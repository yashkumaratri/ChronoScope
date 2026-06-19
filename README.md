# ChronoScope: Evaluating Temporal Consistency in Multi-Turn Language Models

[![Paper](https://img.shields.io/badge/ACL-2026-blue.svg)](https://arxiv.org/abs/your-paper-link) 
[![Dataset](https://img.shields.io/badge/Dataset-ChronoScope-green.svg)](https://huggingface.co/datasets/your-dataset-link)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation and dataset for the paper: **"Evaluating Temporal Consistency in Multi-Turn Language Models"** (ACL 2026).

---

## 📌 Overview

As Large Language Models (LLMs) are increasingly deployed in interactive settings, they must maintain **Temporal Scope Stability**—the ability to preserve, override, or transfer time-scoped factual context across dialogue turns. 

**ChronoScope** is a large-scale diagnostic benchmark designed to test this ability. It consists of over **1.4 million** deterministically generated question chains grounded in **Wikidata**, spanning various domains like politics, sports, and business.

### Key Contributions
* **Benchmark**: A comprehensive dataset for multi-turn temporal question answering.
* **Metric Framework**: New metrics for evaluating **Strict Chain Consistency** and **Temporal Drift**.
* **Analysis**: Extensive evaluation showing that SOTA models (GPT-4o, Gemini 1.5 Pro, Llama 3) frequently drift toward present-day assumptions, even when a historical context was correctly established.

---

## 🛠 Dataset Structure

ChronoScope organizes interactions into **11 distinct chain families** to isolate specific patterns of temporal reasoning:

| Chain Family | Description |
| :--- | :--- |
| **Carryover** | Inheriting scope from a previous turn (e.g., "Who was CEO in 1998?" → "Who was the CFO?"). |
| **Scope Switch** | Explicitly overriding a previous temporal frame with a new one. |
| **Cross-Entity** | Shifting focus to a related entity while maintaining the same year. |
| **Multi-Turn** | Longer chains (3-6 turns) testing cumulative stability. |
| **Temporal Narrative**| Simulating chronological story-tracking across multiple years. |

---

## 🚀 Evaluation Settings

We evaluate models under three distinct context settings:

1.  **Gold Context (Oracle)**: Prior assistant responses are replaced with gold answers to isolate reasoning from error propagation.
2.  **Self-Conditioned**: The model relies on its own previous answers, reflecting realistic interactive usage.
3.  **Questions Only**: Each question is answered in isolation to provide a baseline for implicit vs. explicit cues.

---

## 📊 Results at a Glance

Our findings reveal a significant gap between models' static knowledge and their ability to maintain temporal scope:

* **Temporal Drift**: Models exhibit a strong bias toward the "present," often forgetting the temporal constraints established in Turn 1 by the time they reach Turn 3.
* **Consistency Collapse**: In self-conditioned settings, chain-level accuracy drops significantly compared to turn-level accuracy.

---

## 📂 Repository Structure

```text
├── source/
│   ├── hf_scope_benchmark.py     # Main Hugging Face model evaluation script
│   ├── run_benchmark.sbatch      # SLURM cluster submission template
│   ├── build_label_db.py         # Utilities to build label databases
│   ├── build_stage2_truth.py     # Stage 2 ground truth construction
│   ├── build_stage3_dataset.py   # Final dataset preparation pipeline
│   ├── extract_stage1.py         # Data extraction pipeline (Stage 1)
│   └── extract_from_shards.py    # Shard processing utilities
├── LICENSE
└── README.md
```

## ⚙️ Installation & Usage


To evaluate a model on the **ChronoScope** benchmark.

# Download the raw benchmark file using huggingface-cli
huggingface-cli download yashkumaratri/ChronoScope merged_scope_benchmark.jsonl --repo-type dataset --local-dir /path/to/your/storage/

# Choose execution mode

# Option A: Local / Interactive Execution

python source/hf_scope_benchmark.py \
  --data "/path/to/your/storage/merged_scope_benchmark.jsonl" \
  --model "nvidia/Nemotron-Cascade-8B" \
  --out "results/nemotron_results.json" \
  --max_chains 50000 \
  --self_max_chains 10000 \
  --batch_size 16 \
  --max_new_tokens 24 \
  --dtype bfloat16 \
  --scope_turns_only \
  --match_mode relaxed

# Option B: HPC Cluster Deployment (SLURM)
Make changes in the slurm file and execute

sbatch source/run_benchmark.sbatch



## 📝 Citation

If you find our work or the ChronoScope dataset useful in your research, please cite our paper:
```bibtex
@inproceedings{atri2026chronoscope,
  title={Evaluating Temporal Consistency in Multi-Turn Language Models},
  author={Atri, Yash Kumar and Johnson, Steven L. and Hartvigsen, Tom},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```
**Yash Kumar Atri** — [atri@virginia.edu](mailto:atri@virginia.edu)

