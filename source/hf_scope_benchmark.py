import argparse
import json
import random
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


import os

HF_CACHE = "/sfs/weka/scratch/bkr3as/hf_cache"
HF_HOME = "/sfs/weka/scratch/bkr3as/hf_home"

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = "/sfs/weka/scratch/bkr3as/hf_datasets"
os.environ["HF_MODULES_CACHE"] = "/sfs/weka/scratch/bkr3as/hf_modules"
os.environ["PYTHONPYCACHEPREFIX"] = "/sfs/weka/scratch/bkr3as/py_cache"
os.environ["TMPDIR"] = "/sfs/weka/scratch/bkr3as/tmp"

ALLOWED_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-7b-it",
    "deepseek-ai/deepseek-llm-7b-chat",
    "openai/gpt-oss-20b",
    "microsoft/Phi-3-mini-4k-instruct",
    "nvidia/Nemotron-Cascade-8B",
}


# ============================================================
# Normalization + matching
# ============================================================
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"“": '"', "”": '"', "’": "'", "‘": "'"})

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.translate(_QUOTES)
    s = s.strip().lower()
    s = _WS.sub(" ", s)
    return s

def postprocess_pred(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()

    low = s.lower().strip()
    prefixes = ["assistant:", "answer:", "final answer:", "the answer is", "it is", "it's"]
    for p in prefixes:
        if low.startswith(p):
            s = s[len(p):].strip(" :\t")
            break

    s = s.strip().strip(string.punctuation + " ")
    return s

def normalize_for_match(s: str) -> str:
    s = postprocess_pred(s)
    s = normalize_text(s)
    s = re.sub(r"[^\w\s]", "", s)
    s = _WS.sub(" ", s).strip()
    return s

def extract_candidate_answers(pred: str) -> List[str]:
    if pred is None:
        return []
    s = str(pred).strip()
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()

    m = re.search(r"(?i)\banswer\s*:\s*(.+)$", s)
    if m:
        s = m.group(1).strip()

    s = s.strip().strip('"\'' + string.punctuation + " ")
    cands = [s]

    if "(" in s and ")" in s:
        no_paren = re.sub(r"\s*\([^)]*\)", "", s).strip()
        if no_paren and no_paren != s:
            cands.append(no_paren)

    if "," in s:
        first = s.split(",", 1)[0].strip()
        if first and first != s:
            cands.append(first)

    out, seen = [], set()
    for x in cands:
        nx = normalize_for_match(x)
        if nx and nx not in seen:
            out.append(x)
            seen.add(nx)
    return out

def match_exact(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)

def match_relaxed(pred: str, gold: str) -> bool:
    p = normalize_for_match(pred)
    g = normalize_for_match(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p or p in g:
        return True

    raw = postprocess_pred(pred)
    for sep in [",", ";", "/", "|"]:
        if sep in raw:
            parts = [normalize_for_match(x) for x in raw.split(sep)]
            if g in parts:
                return True
    return False

def tokens(s: str) -> List[str]:
    s = normalize_for_match(s)
    return s.split() if s else []

def token_f1_micro_add(bucket: dict, pred: str, gold: str):
    pt = tokens(pred)
    gt = tokens(gold)
    pc = Counter(pt)
    gc = Counter(gt)
    common = sum((pc & gc).values())
    bucket["tok_tp"] += common
    bucket["tok_pred"] += len(pt)
    bucket["tok_gold"] += len(gt)

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = normalize_for_match(s).replace(" ", "")
    if not s:
        return []
    if len(s) < n:
        return [s]
    return [s[i:i+n] for i in range(len(s) - n + 1)]

def char_f1_micro_add(bucket: dict, pred: str, gold: str, n: int = 3):
    pn = char_ngrams(pred, n=n)
    gn = char_ngrams(gold, n=n)
    pc = Counter(pn)
    gc = Counter(gn)
    common = sum((pc & gc).values())
    bucket["ch_tp"] += common
    bucket["ch_pred"] += len(pn)
    bucket["ch_gold"] += len(gn)


# ============================================================
# Present-day extraction for drift
# ============================================================
def gold_present_value(chain_present: Any, pid: Optional[str] = None) -> Optional[str]:
    if chain_present is None:
        return None
    if isinstance(chain_present, str):
        return chain_present
    if isinstance(chain_present, dict):
        if pid and pid in chain_present and isinstance(chain_present[pid], str):
            return chain_present[pid]
        vals = [v for v in chain_present.values() if isinstance(v, str)]
        if len(vals) == 1:
            return vals[0]
    return None


# ============================================================
# Sampling
# ============================================================
def stratified_sample(chains: List[dict], max_chains: int, seed: int = 42) -> List[dict]:
    random.seed(seed)
    buckets = defaultdict(list)
    for c in chains:
        fam = c.get("family", "unknown")
        L = len(c.get("turns", []) or [])
        if L >= 2:
            buckets[(fam, L)].append(c)

    keys = list(buckets.keys())
    random.shuffle(keys)
    if not keys:
        return []

    per_bucket = max(1, max_chains // len(keys))
    sampled = []
    for k in keys:
        b = buckets[k]
        n = min(per_bucket, len(b))
        sampled.extend(random.sample(b, n))

    if len(sampled) < max_chains:
        remaining = []
        sampled_ids = set(id(x) for x in sampled)
        for c in chains:
            if id(c) not in sampled_ids and len(c.get("turns", []) or []) >= 2:
                remaining.append(c)
        need = max_chains - len(sampled)
        if remaining and need > 0:
            sampled.extend(random.sample(remaining, min(need, len(remaining))))

    if len(sampled) > max_chains:
        sampled = random.sample(sampled, max_chains)

    return sampled


# ============================================================
# Prompting
# ============================================================
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers factual questions.\n"
    "Important:\n"
    "- Use the conversation context.\n"
    "- If a prior turn establishes a time period, keep that same time scope unless it is explicitly changed.\n"
    "- Answer with only the entity or value, no extra explanation.\n"
)

def build_messages(chain: dict, turn_idx: int, context_mode: str, model_answers: Optional[List[str]] = None) -> List[Dict[str, str]]:
    turns = chain["turns"]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(turn_idx):
        msgs.append({"role": "user", "content": turns[i]["question"]})

        if context_mode == "gold":
            msgs.append({"role": "assistant", "content": str(turns[i]["answer"])})
        elif context_mode == "self":
            assert model_answers is not None and i < len(model_answers)
            msgs.append({"role": "assistant", "content": str(model_answers[i])})
        elif context_mode == "questions_only":
            pass
        else:
            raise ValueError(f"Unknown context_mode={context_mode}")

    msgs.append({"role": "user", "content": turns[turn_idx]["question"]})
    return msgs

def messages_to_prompt(messages: List[Dict[str, str]], tokenizer) -> str:
    # Check role alternation (ignore initial system)
    roles = [m["role"] for m in messages if m["role"] != "system"]
    alternates = all(
        roles[i] != roles[i + 1]
        for i in range(len(roles) - 1)
    )

    # Use chat template only if valid
    if alternates and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass  # fall back safely

    # Fallback: plain text prompt (model-agnostic)
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Assistant: {m['content']}")

    parts.append("Assistant:")
    return "\n".join(parts)



# ============================================================
# Optional semantic matching
# ============================================================
class SemanticMatcher:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.device = device
        self.backend = None

        try:
            from sentence_transformers import SentenceTransformer
            self.backend = "st"
            self.model = SentenceTransformer(model_name, device=device)
        except Exception:
            self.backend = "hf"
            import torch
            from transformers import AutoTokenizer, AutoModel

            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.enc = AutoModel.from_pretrained(model_name)
            self.enc.to(device)
            self.enc.eval()

    def embed(self, texts: List[str]):
        if self.backend == "st":
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        import torch
        with torch.no_grad():
            batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            out = self.enc(**batch)
            last = out.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.detach().cpu().numpy()

    @staticmethod
    def cosine(a, b) -> float:
        return float((a * b).sum())

    def is_semantic_match(self, pred: str, gold: str, thr: float) -> bool:
        p = postprocess_pred(pred)
        g = str(gold)
        if not p or not g:
            return False
        vecs = self.embed([p, g])
        return self.cosine(vecs[0], vecs[1]) >= thr


# ============================================================
# Fast batched generate() runner (no pipeline)
# ============================================================
@dataclass
class GenConfig:
    max_new_tokens: int = 24
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 16
    answer_k: int = 1
    num_beams: int = 1
    dtype: str = "bfloat16"  # float16 / bfloat16 / auto
    device: str = "cuda"

class FastHFGenerator:
    def __init__(self, model_name: str, cfg: GenConfig):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.model_name = model_name
        self.device = cfg.device

        self.tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Batched decoder-only generation tends to be faster with left padding
        self.tokenizer.padding_side = "left"

        torch_dtype = None
        if cfg.dtype == "float16":
            torch_dtype = torch.float16
        elif cfg.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=HF_CACHE,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    device_map={"": self.device},
)

        self.model.eval()
        self.model.requires_grad_(False)
        
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    def generate_topk_batch(self, prompts: List[str], cfg: GenConfig) -> List[List[str]]:
        torch = self.torch
        do_sample = cfg.temperature > 0.0
        k = max(1, cfg.answer_k)
        num_beams = max(cfg.num_beams, k)
    
        results: List[List[str]] = [[] for _ in range(len(prompts))]
        bs = max(1, cfg.batch_size)
    
        for i in range(0, len(prompts), bs):
            batch_prompts = prompts[i:i+bs]
    
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
    
            input_length = enc.input_ids.shape[1]  # <-- ADD THIS
    
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=do_sample,
                    temperature=(cfg.temperature if do_sample else None),
                    top_p=cfg.top_p,
                    num_beams=num_beams,
                    num_return_sequences=k,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # 🔧 Phi-3 compatibility fix
                if "phi-3" in self.model_name.lower():
                    gen_kwargs["use_cache"] = False
                
                with torch.inference_mode():
                    gen_ids = self.model.generate(
                        **enc,
                        **gen_kwargs,
                    )

    
            # Only decode the NEW tokens (strip input)
            new_tokens = gen_ids[:, input_length:]  # <-- ADD THIS
            decoded_new = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)  # <-- CHANGE THIS
    
            # Group by input
            if k == 1:
                for j, cont in enumerate(decoded_new):
                    results[i + j] = [cont.strip()]
            else:
                for j in range(len(batch_prompts)):
                    cands = []
                    for r in range(k):
                        cont = decoded_new[j*k + r].strip()
                        cands.append(cont)
                    results[i + j] = cands
    
        return results


# ============================================================
# Metrics buckets
# ============================================================
def new_bucket():
    return {
        "n_chains": 0,
        "n_turns": 0,
        "correct_at1": 0,
        "correct_atk": 0,

        "tok_tp": 0, "tok_pred": 0, "tok_gold": 0,
        "ch_tp": 0, "ch_pred": 0, "ch_gold": 0,

        "strict_chain_at1": 0,
        "avg_chain_turn_acc_at1_sum": 0.0,
        "final_turn_acc_at1_sum": 0.0,
        "final_turn_acc_atk_sum": 0.0,

        "n_present_available": 0,
        "n_drift_at1": 0,
        "n_drift_atk": 0,
        "n_errors_at1": 0,
    }

def finalize(bucket: dict) -> dict:
    nT = max(1, bucket["n_turns"])
    nC = max(1, bucket["n_chains"])

    acc1 = bucket["correct_at1"] / nT
    acck = bucket["correct_atk"] / nT

    tok_p = bucket["tok_tp"] / max(1, bucket["tok_pred"])
    tok_r = bucket["tok_tp"] / max(1, bucket["tok_gold"])
    tok_f1 = (2 * tok_p * tok_r / (tok_p + tok_r)) if (tok_p + tok_r) else 0.0

    ch_p = bucket["ch_tp"] / max(1, bucket["ch_pred"])
    ch_r = bucket["ch_tp"] / max(1, bucket["ch_gold"])
    ch_f1 = (2 * ch_p * ch_r / (ch_p + ch_r)) if (ch_p + ch_r) else 0.0

    present = bucket["n_present_available"]
    drift1 = bucket["n_drift_at1"]
    driftk = bucket["n_drift_atk"]
    err1 = bucket["n_errors_at1"]

    return {
        "n_chains": bucket["n_chains"],
        "n_turns": bucket["n_turns"],
        "acc_at1": acc1,
        "acc_atk": acck,
        "token_f1_micro_top1": tok_f1,
        "char3_f1_micro_top1": ch_f1,
        "strict_chain_acc_at1": bucket["strict_chain_at1"] / nC,
        "avg_chain_turn_acc_at1": bucket["avg_chain_turn_acc_at1_sum"] / nC,
        "final_turn_acc_at1": bucket["final_turn_acc_at1_sum"] / nC,
        "final_turn_acc_atk": bucket["final_turn_acc_atk_sum"] / nC,
        "present_available_rate": present / max(1, nT),
        "drift_rate_given_present_at1": (drift1 / present) if present else 0.0,
        "drift_rate_given_present_atk": (driftk / present) if present else 0.0,
        "drift_rate_given_error_at1": (drift1 / err1) if err1 else 0.0,
    }


# ============================================================
# Load JSONL
# ============================================================
def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


# ============================================================
# Matching function builder
# ============================================================
def make_is_match(match_mode: str, semantic: Optional[SemanticMatcher], semantic_thr: float):
    def base_match(pred: str, gold: str) -> bool:
        return match_exact(pred, gold) if match_mode == "exact" else match_relaxed(pred, gold)

    def is_match(pred: str, gold: str) -> bool:
        cands = extract_candidate_answers(pred) or [pred]
        for c in cands:
            if base_match(c, gold):
                return True
        if semantic is not None:
            for c in cands:
                if semantic.is_semantic_match(c, gold, semantic_thr):
                    return True
        return False
    return is_match


# ============================================================
# Evaluate (gold/questions_only) in turn-batched style
# ============================================================
def evaluate_batched_mode(
    gen: FastHFGenerator,
    chains: List[dict],
    cfg: GenConfig,
    start_idx: int,
    context_mode: str,
    is_match,
    max_bad: int,
):
    prompt_cache = {}
    by_family = defaultdict(new_bucket)
    by_length = defaultdict(new_bucket)
    by_pos = defaultdict(new_bucket)
    overall = new_bucket()
    bad_samples = []

    items = []
    last_eval_turn = {}
    for ci, chain in enumerate(chains):
        turns = chain.get("turns", []) or []
        L = len(turns)
        if L < 2:
            continue
        last_eval_turn[ci] = L - 1
        for ti in range(start_idx, L):
            items.append((ci, ti))

    chain_eval = defaultdict(int)
    chain_ok1 = defaultdict(int)
    chain_err1 = defaultdict(int)
    chain_last_ok1 = defaultdict(int)
    chain_last_okk = defaultdict(int)

    for i in tqdm(range(0, len(items), cfg.batch_size), desc=f"Evaluating turns ({context_mode})"):
        batch = items[i:i+cfg.batch_size]
        prompts, metas = [], []

        for ci, ti in batch:
            chain = chains[ci]
        
            key = (ci, ti, context_mode)
            if key not in prompt_cache:
                msgs = build_messages(chain, ti, context_mode=context_mode)
                prompt_cache[key] = messages_to_prompt(msgs, gen.tokenizer)
        
            prompt = prompt_cache[key]
            prompts.append(prompt)
            metas.append((ci, ti))


        topk = gen.generate_topk_batch(prompts, cfg)

        for (ci, ti), preds in zip(metas, topk):
            chain = chains[ci]
            fam = chain.get("family", "unknown")
            turns = chain["turns"]
            L = len(turns)
            pos = (ti - start_idx) + 1

            gold = str(turns[ti]["answer"])
            pid = turns[ti].get("pid")
            present = gold_present_value(chain.get("present_day_answer"), pid=pid)

            pred1 = preds[0] if preds else ""
            ok1 = is_match(pred1, gold)
            okk = any(is_match(p, gold) for p in preds)

            drift1 = False
            driftk = False
            if present:
                drift1 = (not ok1) and is_match(pred1, present)
                driftk = (not okk) and any(is_match(p, present) for p in preds)

            for bucket in (overall, by_family[fam], by_length[L], by_pos[pos]):
                bucket["n_turns"] += 1
                bucket["correct_at1"] += int(ok1)
                bucket["correct_atk"] += int(okk)

                token_f1_micro_add(bucket, pred1, gold)
                char_f1_micro_add(bucket, pred1, gold, n=3)

                if present:
                    bucket["n_present_available"] += 1
                    bucket["n_drift_at1"] += int(drift1)
                    bucket["n_drift_atk"] += int(driftk)

                if not ok1:
                    bucket["n_errors_at1"] += 1

            chain_eval[ci] += 1
            chain_ok1[ci] += int(ok1)
            if not ok1:
                chain_err1[ci] += 1

            if ti == last_eval_turn.get(ci, -1):
                chain_last_ok1[ci] = int(ok1)
                chain_last_okk[ci] = int(okk)

            if ok1 and len(bad_samples) < max_bad // 4:
                        bad_samples.append({
                            "chain_id": chain.get("chain_id"),
                            "family": fam,
                            "length": L,
                            "turn_index": ti,
                            "followup_pos": pos,
                            "question": turns[ti]["question"],
                            "gold": gold,
                            "pred_top1": pred1,
                            "present_day": present,
                            "drift_top1": False,
                            "drift_topk": False,
                            "error_type": "correct",
                        })


            if (not ok1) and len(bad_samples) < max_bad:
                bad_samples.append({
                    "chain_id": chain.get("chain_id"),
                    "family": fam,
                    "length": L,
                    "turn_index": ti,
                    "followup_pos": pos,
                    "question": turns[ti]["question"],
                    "gold": gold,
                    "pred_top1": pred1,
                    "pred_topk": preds[: min(5, len(preds))],
                    "present_day": present,
                    "drift_top1": drift1,
                    "drift_topk": driftk,
                    "error_type": (
                                    "drift"
                                    if drift1
                                    else "hallucination"
                                    if not ok1 and not present
                                    else "wrong_answer"
                                ),

                })

    # chain finalize
    for ci, chain in enumerate(chains):
        turns = chain.get("turns", []) or []
        L = len(turns)
        if L < 2:
            continue
        fam = chain.get("family", "unknown")
        n_eval = chain_eval.get(ci, 0)
        if n_eval == 0:
            continue

        strict_ok = (chain_err1.get(ci, 0) == 0)
        frac_ok = chain_ok1.get(ci, 0) / n_eval

        for bucket in (overall, by_family[fam], by_length[L]):
            bucket["n_chains"] += 1
            bucket["strict_chain_at1"] += int(strict_ok)
            bucket["avg_chain_turn_acc_at1_sum"] += frac_ok
            bucket["final_turn_acc_at1_sum"] += chain_last_ok1.get(ci, 0)
            bucket["final_turn_acc_atk_sum"] += chain_last_okk.get(ci, 0)

    return {
    "overall": finalize(overall),
    "by_family": {k: finalize(v) for k, v in by_family.items()},
    "by_length": {int(k): finalize(v) for k, v in by_length.items()},
    "by_followup_pos": {int(k): finalize(v) for k, v in by_pos.items()},
    "examples": bad_samples,
}





# ============================================================
# Evaluate self-conditioned (chain sequential, but uses fast generate)
# ============================================================
def evaluate_self_mode(
    gen: FastHFGenerator,
    chains: List[dict],
    cfg: GenConfig,
    start_idx: int,
    is_match,
    max_bad: int,
):
    cfg.answer_k = 1
    cfg.num_beams = 1
    cfg.temperature = 0.0

    by_family = defaultdict(new_bucket)
    by_length = defaultdict(new_bucket)
    by_pos = defaultdict(new_bucket)
    overall = new_bucket()
    bad_samples = []

    last_eval_turn = {ci: (len(c.get("turns", []) or []) - 1) for ci, c in enumerate(chains)}

    chain_eval = defaultdict(int)
    chain_ok1 = defaultdict(int)
    chain_err1 = defaultdict(int)
    chain_last_ok1 = defaultdict(int)
    chain_last_okk = defaultdict(int)

    for ci, chain in tqdm(list(enumerate(chains)), desc="Evaluating chains (self)"):
        turns = chain.get("turns", []) or []
        L = len(turns)
        if L < 2:
            continue

        model_answers: List[str] = []

        for ti in range(L):
            msgs = build_messages(chain, ti, context_mode="self", model_answers=model_answers)
            prompt = messages_to_prompt(msgs, gen.tokenizer)
            preds = gen.generate_topk_batch([prompt], cfg)[0]
            pred1 = preds[0] if preds else ""
            model_answers.append(pred1)

            if ti < start_idx:
                continue

            fam = chain.get("family", "unknown")
            pos = (ti - start_idx) + 1

            gold = str(turns[ti]["answer"])
            pid = turns[ti].get("pid")
            present = gold_present_value(chain.get("present_day_answer"), pid=pid)

            ok1 = is_match(pred1, gold)
            okk = any(is_match(p, gold) for p in preds)

            drift1 = False
            driftk = False
            if present:
                drift1 = (not ok1) and is_match(pred1, present)
                driftk = (not okk) and any(is_match(p, present) for p in preds)

            for bucket in (overall, by_family[fam], by_length[L], by_pos[pos]):
                bucket["n_turns"] += 1
                bucket["correct_at1"] += int(ok1)
                bucket["correct_atk"] += int(okk)

                token_f1_micro_add(bucket, pred1, gold)
                char_f1_micro_add(bucket, pred1, gold, n=3)

                if present:
                    bucket["n_present_available"] += 1
                    bucket["n_drift_at1"] += int(drift1)
                    bucket["n_drift_atk"] += int(driftk)

                if not ok1:
                    bucket["n_errors_at1"] += 1

            chain_eval[ci] += 1
            chain_ok1[ci] += int(ok1)
            if not ok1:
                chain_err1[ci] += 1

            if ti == last_eval_turn.get(ci, -1):
                chain_last_ok1[ci] = int(ok1)
                chain_last_okk[ci] = int(okk)

            if (not ok1) and len(bad_samples) < max_bad:
                bad_samples.append({
                    "chain_id": chain.get("chain_id"),
                    "family": fam,
                    "length": L,
                    "turn_index": ti,
                    "followup_pos": pos,
                    "question": turns[ti]["question"],
                    "gold": gold,
                    "pred_top1": pred1,
                    "pred_topk": preds[: min(5, len(preds))],
                    "present_day": present,
                    "drift_top1": drift1,
                    "drift_topk": driftk,
                    "error_type": (
                                "drift"
                                if drift1
                                else "hallucination"
                                if not ok1 and not present
                                else "wrong_answer"
                            ),

                })

    # chain finalize
    for ci, chain in enumerate(chains):
        turns = chain.get("turns", []) or []
        L = len(turns)
        if L < 2:
            continue
        fam = chain.get("family", "unknown")
        n_eval = chain_eval.get(ci, 0)
        if n_eval == 0:
            continue

        strict_ok = (chain_err1.get(ci, 0) == 0)
        frac_ok = chain_ok1.get(ci, 0) / n_eval

        for bucket in (overall, by_family[fam], by_length[L]):
            bucket["n_chains"] += 1
            bucket["strict_chain_at1"] += int(strict_ok)
            bucket["avg_chain_turn_acc_at1_sum"] += frac_ok
            bucket["final_turn_acc_at1_sum"] += chain_last_ok1.get(ci, 0)
            bucket["final_turn_acc_atk_sum"] += chain_last_okk.get(ci, 0)

    return {
        "overall": finalize(overall),
        "by_family": {k: finalize(v) for k, v in by_family.items()},
        "by_length": {int(k): finalize(v) for k, v in by_length.items()},
        "by_followup_pos": {int(k): finalize(v) for k, v in by_pos.items()},
        "bad_samples": bad_samples,
    }


# ============================================================
# TSV writer (optional)
# ============================================================
def append_rows_tsv(summary_tsv: Path, model: str, ctx_results: Dict[str, dict], cfg_args: dict):
    header = [
        "model", "context", "k", "n_chains", "n_turns",
        "acc@1", "acc@k", "tok_f1", "char3_f1",
        "final@1", "final@k",
        "drift@1|present", "drift@k|present",
    ]
    write_header = not summary_tsv.exists()
    with summary_tsv.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        for ctx, res in ctx_results.items():
            o = res["overall"]
            row = [
                model,
                ctx,
                str(cfg_args.get("answer_k", 1)),
                str(o.get("n_chains", "")),
                str(o.get("n_turns", "")),
                f"{o.get('acc_at1', 0.0):.6f}",
                f"{o.get('acc_atk', 0.0):.6f}",
                f"{o.get('token_f1_micro_top1', 0.0):.6f}",
                f"{o.get('char3_f1_micro_top1', 0.0):.6f}",
                f"{o.get('final_turn_acc_at1', 0.0):.6f}",
                f"{o.get('final_turn_acc_atk', 0.0):.6f}",
                f"{o.get('drift_rate_given_present_at1', 0.0):.6f}",
                f"{o.get('drift_rate_given_present_atk', 0.0):.6f}",
            ]
            f.write("\t".join(row) + "\n")
    print("Appended TSV:", summary_tsv)

def collect_examples(ctx, res):
    if "examples" not in res:
        return []
    out = []
    for ex in res["examples"]:
        ex2 = dict(ex)
        ex2["setting"] = ctx
        out.append(ex2)
    return out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match_mode", type=str, choices=["exact", "relaxed"], default="relaxed")
    ap.add_argument("--scope_turns_only", action="store_true")

    ap.add_argument("--max_chains", type=int, default=5000)
    ap.add_argument("--self_max_chains", type=int, default=1000)

    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--answer_k", type=int, default=1)
    ap.add_argument("--num_beams", type=int, default=1)

    ap.add_argument("--dtype", type=str, choices=["auto", "float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--semantic_model", type=str, default=None)
    ap.add_argument("--semantic_thr", type=float, default=0.80)
    ap.add_argument("--semantic_device", type=str, default="cuda:0")

    ap.add_argument("--max_bad_samples", type=int, default=60)
    ap.add_argument("--summary_tsv", type=str, default=None)
    ap.add_argument("--dump_examples", action="store_true",
                    help="Save per-turn examples for qualitative analysis")


    args = ap.parse_args()
    random.seed(args.seed)

    if args.model not in ALLOWED_MODELS:
        raise ValueError(f"Model {args.model} not in finalized benchmark set")


    chains_all = load_jsonl(Path(args.data))
    sampled = stratified_sample(chains_all, max_chains=args.max_chains, seed=args.seed)
    self_subset = sampled[: min(args.self_max_chains, len(sampled))]

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        answer_k=max(1, args.answer_k),
        num_beams=max(1, args.num_beams),
        dtype=args.dtype,
        device=args.device,
    )

    gen = FastHFGenerator(args.model, cfg)

    semantic = None
    if args.semantic_model:
        semantic = SemanticMatcher(args.semantic_model, device=args.semantic_device)

    is_match = make_is_match(args.match_mode, semantic, args.semantic_thr)

    start_idx = 1 if args.scope_turns_only else 0

    all_results = {}
    t0 = time.time()

    all_results["gold"] = evaluate_batched_mode(
        gen=gen,
        chains=sampled,
        cfg=cfg,
        start_idx=start_idx,
        context_mode="gold",
        is_match=is_match,
        max_bad=args.max_bad_samples,
    )

    all_results["questions_only"] = evaluate_batched_mode(
        gen=gen,
        chains=sampled,
        cfg=cfg,
        start_idx=start_idx,
        context_mode="questions_only",
        is_match=is_match,
        max_bad=args.max_bad_samples,
    )

    all_results["self"] = evaluate_self_mode(
        gen=gen,
        chains=self_subset,
        cfg=cfg,
        start_idx=start_idx,
        is_match=is_match,
        max_bad=args.max_bad_samples,
    )

    examples = []
    for ctx in ["gold", "questions_only", "self"]:
        for ex in all_results[ctx].get("examples", []):
            ex = dict(ex)
            ex["setting"] = ctx
            examples.append(ex)


    dt = time.time() - t0
    all_examples = []

    if args.dump_examples:
        all_examples.extend(collect_examples("gold", all_results["gold"]))
        all_examples.extend(collect_examples("questions_only", all_results["questions_only"]))
        all_examples.extend(collect_examples("self", all_results["self"]))

    payload = {
    "model": args.model,
    "data": args.data,
    "seed": args.seed,
    "sampled_chains_gold_qonly": len(sampled),
    "sampled_chains_self": len(self_subset),
    "args": vars(args),
    "results": all_results,
    "examples": examples,
    "runtime_seconds": dt,
}

    
    if args.dump_examples:
        payload["examples"] = all_examples





    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved:", out_path)
    print("Runtime(s):", round(dt, 2))

    for ctx in ["gold", "questions_only", "self"]:
        o = all_results[ctx]["overall"]
        print(f"\n=== {ctx} ===")
        print("acc@1:", o["acc_at1"])
        print("acc@k:", o["acc_atk"])
        print("final@1:", o["final_turn_acc_at1"])
        print("final@k:", o["final_turn_acc_atk"])
        print("drift@1|present:", o["drift_rate_given_present_at1"])
        print("drift@k|present:", o["drift_rate_given_present_atk"])

    if args.summary_tsv:
        append_rows_tsv(Path(args.summary_tsv), args.model, all_results, vars(args))


if __name__ == "__main__":
    main()
