#!/usr/bin/env python3
# ============================================================
# Stage 3: QA + CHAIN Generation from Wikidata Temporal Truth Tables
# Outputs:
#   - data.jsonl   : self-contained single-turn QAs
#   - chains.jsonl : multi-turn chains with turns[]
# ============================================================

import json
import random
import uuid
import re
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Iterable, Set, Tuple, Any

import polars as pl
from tqdm import tqdm

import argparse
import json
import random
import time
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple
from collections import defaultdict, Counter

import polars as pl
from tqdm import tqdm
from typing import Optional


# -----------------------------
# Property specs and complements
# -----------------------------

PID_SPECS: Dict[str, Dict[str, str]] = {
    "P35": {"qtype": "who",   "slot": "head_of_state",         "surface": "head of state"},
    "P6":  {"qtype": "who",   "slot": "head_of_government",    "surface": "head of government"},
    "P39": {"qtype": "what",  "slot": "position_held",         "surface": "position held"},
    "P102":{"qtype": "which", "slot": "political_party",       "surface": "political party"},

    "P108":{"qtype": "which", "slot": "employer",              "surface": "employer"},
    "P463":{"qtype": "which", "slot": "member_of",             "surface": "member of"},
    "P127":{"qtype": "which", "slot": "owned_by",              "surface": "owned by"},
    "P169":{"qtype": "who",   "slot": "chief_executive",       "surface": "chief executive officer"},
    "P488":{"qtype": "who",   "slot": "chairperson",           "surface": "chairperson"},

    "P69": {"qtype": "which", "slot": "educated_at",           "surface": "educated at"},
    "P106":{"qtype": "what",  "slot": "occupation",            "surface": "occupation"},
    "P101":{"qtype": "what",  "slot": "field_of_work",         "surface": "field of work"},

    "P131":{"qtype": "where", "slot": "located_in_admin",      "surface": "located in the administrative entity"},
    "P17": {"qtype": "which", "slot": "country",               "surface": "country"},
    "P276":{"qtype": "where", "slot": "location",              "surface": "location"},

    "P54": {"qtype": "which", "slot": "sports_team",           "surface": "sports team"},
    "P286":{"qtype": "who",   "slot": "head_coach",            "surface": "head coach"},

    "P31": {"qtype": "what",  "slot": "instance_of",           "surface": "instance of"},
    "P279":{"qtype": "what",  "slot": "subclass_of",           "surface": "subclass of"},
}

PID_REALIZATION: Dict[str, Dict[str, List[str]]] = {
    "P35": {
        "explicit_temporal": [
            "In {year}, who was the head of state of {subj}?",
            "Who was {subj}'s head of state in {year}?",
            "Who served as head of state of {subj} in {year}?",
        ],
        "explicit_timeless": [
            "Who is the head of state of {subj}?",
            "Who serves as head of state of {subj}?",
        ],
        "followup": ["Who was the head of state?", "Who held that office?"],
        "followup_then": ["Who was the head of state then?", "Who held that office at the time?"],
        "switch": ["In {year2}, who was the head of state?", "What about in {year2}: who was the head of state?"],
    },
    "P6": {
        "explicit_temporal": [
            "In {year}, who was the head of government of {subj}?",
            "Who was {subj}'s head of government in {year}?",
            "Who served as head of government of {subj} in {year}?",
        ],
        "explicit_timeless": [
            "Who is the head of government of {subj}?",
            "Who serves as head of government of {subj}?",
        ],
        "followup": ["Who was the head of government?", "Who held that office?"],
        "followup_then": ["Who was the head of government then?", "Who held that office at the time?"],
        "switch": ["In {year2}, who was the head of government?", "What about in {year2}: who was head of government?"],
    },
    "P39": {
        "explicit_temporal": [
            "What position did {subj} hold in {year}?",
            "In {year}, what office or position did {subj} hold?",
            "Which position did {subj} hold in {year}?",
        ],
        "explicit_timeless": [
            "What position does {subj} hold?",
            "Which office or position does {subj} hold?",
        ],
        "followup": ["What position was it?", "Which position was that?"],
        "followup_then": ["What position was it then?", "Which position was that at the time?"],
        "switch": ["In {year2}, what position did they hold?", "What about in {year2}: what position did they hold?"],
    },
    "P102": {
        "explicit_temporal": [
            "In {year}, which political party was {subj} affiliated with?",
            "What political party was {subj} part of in {year}?",
            "Which party did {subj} belong to in {year}?",
        ],
        "explicit_timeless": [
            "Which political party is {subj} affiliated with?",
            "What political party is {subj} part of?",
        ],
        "followup": ["Which political party was it?", "What party was that?"],
        "followup_then": ["Which political party was it then?", "What party was that at the time?"],
        "switch": ["In {year2}, which political party was it?", "What about in {year2}: which party was it?"],
    },
    "P108": {
        "explicit_temporal": [
            "Where did {subj} work in {year}?",
            "Who employed {subj} in {year}?",
            "Which organization did {subj} work for in {year}?",
        ],
        "explicit_timeless": [
            "Where does {subj} work?",
            "Who employs {subj}?",
        ],
        "followup": ["Where did they work?", "Who employed them?"],
        "followup_then": ["Where did they work then?", "Who employed them at the time?"],
        "switch": ["In {year2}, where did they work?", "What about in {year2}: who employed them?"],
    },
    "P463": {
        "explicit_temporal": [
            "In {year}, which organization was {subj} a member of?",
            "What organization did {subj} belong to in {year}?",
            "Which group was {subj} a member of in {year}?",
        ],
        "explicit_timeless": [
            "Which organization is {subj} a member of?",
            "What group does {subj} belong to?",
        ],
        "followup": ["Which organization was it a member of?", "What group was it part of?"],
        "followup_then": ["Which organization was it a member of then?", "What group was it part of at the time?"],
        "switch": ["In {year2}, which organization was it a member of?", "What about in {year2}: what group was it part of?"],
    },
    "P127": {
        "explicit_temporal": [
            "Who owned {subj} in {year}?",
            "In {year}, who was {subj} owned by?",
            "Which organization owned {subj} in {year}?",
        ],
        "explicit_timeless": [
            "Who owns {subj}?",
            "Which organization owns {subj}?",
        ],
        "followup": ["Who owned it?", "Which organization owned it?"],
        "followup_then": ["Who owned it then?", "Which organization owned it at the time?"],
        "switch": ["In {year2}, who owned it?", "What about in {year2}: which organization owned it?"],
    },
    "P169": {
        "explicit_temporal": [
            "In {year}, who was the CEO of {subj}?",
            "Who served as CEO of {subj} in {year}?",
            "Who was {subj}'s chief executive officer in {year}?",
        ],
        "explicit_timeless": [
            "Who is the CEO of {subj}?",
            "Who serves as CEO of {subj}?",
        ],
        "followup": ["Who was the CEO?", "Who served as CEO?"],
        "followup_then": ["Who was the CEO then?", "Who served as CEO at the time?"],
        "switch": ["In {year2}, who was the CEO?", "What about in {year2}: who served as CEO?"],
    },
    "P488": {
        "explicit_temporal": [
            "In {year}, who was the chairperson of {subj}?",
            "Who chaired {subj} in {year}?",
            "Who served as chairperson of {subj} in {year}?",
        ],
        "explicit_timeless": [
            "Who is the chairperson of {subj}?",
            "Who chairs {subj}?",
        ],
        "followup": ["Who was the chairperson?", "Who chaired it?"],
        "followup_then": ["Who was the chairperson then?", "Who chaired it at the time?"],
        "switch": ["In {year2}, who was the chairperson?", "What about in {year2}: who chaired it?"],
    },
    "P69": {
        "explicit_temporal": [
            "Where did {subj} study in {year}?",
            "Which school did {subj} attend in {year}?",
            "In {year}, where was {subj} educated?",
        ],
        "explicit_timeless": [
            "Where did {subj} study?",
            "Which school did {subj} attend?",
        ],
        "followup": ["Where did they study?", "Which school did they attend?"],
        "followup_then": ["Where did they study then?", "Which school did they attend at the time?"],
        "switch": ["In {year2}, where did they study?", "What about in {year2}: which school did they attend?"],
    },
    "P106": {
        "explicit_temporal": [
            "What was {subj}'s occupation in {year}?",
            "In {year}, what did {subj} do for a living?",
            "What was {subj}'s job in {year}?",
        ],
        "explicit_timeless": [
            "What is {subj}'s occupation?",
            "What does {subj} do for a living?",
        ],
        "followup": ["What was their occupation?", "What did they do for a living?"],
        "followup_then": ["What was their occupation then?", "What did they do for a living at the time?"],
        "switch": ["In {year2}, what was their occupation?", "What about in {year2}: what did they do for a living?"],
    },
    "P101": {
        "explicit_temporal": [
            "What was {subj}'s field of work in {year}?",
            "In {year}, what field did {subj} work in?",
            "What area did {subj} work in during {year}?",
        ],
        "explicit_timeless": [
            "What is {subj}'s field of work?",
            "What field does {subj} work in?",
        ],
        "followup": ["What was their field of work?", "What field did they work in?"],
        "followup_then": ["What was their field of work then?", "What field did they work in at the time?"],
        "switch": ["In {year2}, what field did they work in?", "What about in {year2}: what was their field of work?"],
    },
    "P131": {
        "explicit_temporal": [
            "In {year}, which administrative area was {subj} located in?",
            "Where within its administrative region was {subj} in {year}?",
            "Which administrative entity contained {subj} in {year}?",
        ],
        "explicit_timeless": [
            "Which administrative area is {subj} located in?",
            "What administrative entity contains {subj}?",
        ],
        "followup": ["Which administrative area was it in?", "Where was it located administratively?"],
        "followup_then": ["Which administrative area was it in then?", "Where was it located administratively at the time?"],
        "switch": ["In {year2}, which administrative area was it in?", "What about in {year2}: where was it located administratively?"],
    },
    "P17": {
        "explicit_temporal": [
            "In {year}, which country was {subj} in?",
            "What country was {subj} in during {year}?",
            "Which country did {subj} belong to in {year}?",
        ],
        "explicit_timeless": [
            "Which country is {subj} in?",
            "What country does {subj} belong to?",
        ],
        "followup": ["Which country was it in?", "What country was that in?"],
        "followup_then": ["Which country was it in then?", "What country was that at the time?"],
        "switch": ["In {year2}, which country was it in?", "What about in {year2}: what country was it in?"],
    },
    "P276": {
        "explicit_temporal": [
            "Where was {subj} located in {year}?",
            "What was the location of {subj} in {year}?",
            "In {year}, where could {subj} be found?",
        ],
        "explicit_timeless": [
            "Where is {subj} located?",
            "What is the location of {subj}?",
        ],
        "followup": ["Where was it located?", "What was its location?"],
        "followup_then": ["Where was it located then?", "What was its location at the time?"],
        "switch": ["In {year2}, where was it located?", "What about in {year2}: where could it be found?"],
    },
    "P54": {
        "explicit_temporal": [
            "In {year}, which team did {subj} play for?",
            "What team was {subj} on in {year}?",
            "Which club did {subj} play for in {year}?",
        ],
        "explicit_timeless": [
            "Which team does {subj} play for?",
            "What team is {subj} on?",
        ],
        "followup": ["Which team was it?", "What team were they on?"],
        "followup_then": ["Which team was it then?", "What team were they on at the time?"],
        "switch": ["In {year2}, which team did they play for?", "What about in {year2}: what team were they on?"],
    },
    "P286": {
        "explicit_temporal": [
            "In {year}, who was the head coach of {subj}?",
            "Who coached {subj} in {year}?",
            "Who served as head coach of {subj} in {year}?",
        ],
        "explicit_timeless": [
            "Who is the head coach of {subj}?",
            "Who coaches {subj}?",
        ],
        "followup": ["Who was the head coach?", "Who coached them?"],
        "followup_then": ["Who was the head coach then?", "Who coached them at the time?"],
        "switch": ["In {year2}, who was the head coach?", "What about in {year2}: who coached them?"],
    },
    "P31": {
        "explicit_temporal": [
            "In {year}, what type of thing was {subj}?",
            "In {year}, what kind of thing was {subj}?",
            "How would you categorize {subj} in {year}?",
        ],
        "explicit_timeless": [
            "What type of thing is {subj}?",
            "What kind of thing is {subj}?",
        ],
        "followup": ["What type of thing was it?", "How would you categorize it?"],
        "followup_then": ["What type of thing was it then?", "How would you categorize it at the time?"],
        "switch": ["In {year2}, what type of thing was it?", "What about in {year2}: how would you categorize it?"],
    },
    "P279": {
        "explicit_temporal": [
            "In {year}, what was {subj} a subclass of?",
            "What broader class did {subj} belong to in {year}?",
            "In {year}, {subj} was a subclass of what?",
        ],
        "explicit_timeless": [
            "What is {subj} a subclass of?",
            "What broader class does {subj} belong to?",
        ],
        "followup": ["What was it a subclass of?", "What broader class did it belong to?"],
        "followup_then": ["What was it a subclass of then?", "What broader class did it belong to at the time?"],
        "switch": ["In {year2}, what was it a subclass of?", "What about in {year2}: what broader class did it belong to?"],
    },
}

BAD_LABELS = {
    "unknown", "unknown value", "none", "null", "n/a", "na",
    "unspecified", "not known", "not applicable",
}

RELATION_FAMILIES: Dict[str, str] = {
    "P35": "leadership",
    "P6": "leadership",
    "P169": "leadership",
    "P488": "leadership",
    "P286": "leadership",
    "P39": "role",
    "P102": "affiliation",
    "P108": "affiliation",
    "P463": "affiliation",
    "P54": "affiliation",
    "P69": "affiliation",
    "P127": "ownership",
    "P17": "geography",
    "P131": "geography",
    "P276": "geography",
    "P31": "ontology",
    "P279": "ontology",
    "P106": "attribute",
    "P101": "attribute",
}

COMPLEMENTS: Dict[str, str] = {
    "P6": "P35", "P35": "P6",
    "P39": "P102", "P102": "P39",
    "P108": "P463", "P463": "P108",
    "P54": "P286", "P286": "P54",
    "P127": "P169", "P169": "P127",
    "P488": "P169",
    "P131": "P17", "P17": "P131",
}

# -----------------------------
# English-safe paraphrases
# Key idea: paraphrase per PID where needed.
# All templates must be self-contained if used in data.jsonl.
# -----------------------------
PARA = {
    "explicit": {
        "who": [
            "Who was the {prop} of {subj} in {year}?",
            "In {year}, who served as the {prop} of {subj}?",
            "Who held the role of {prop} for {subj} in {year}?",
        ],
        "what": [
            "In {year}, what was the {prop} of {subj}?",
            "What {prop} did {subj} have in {year}?",
            "In {year}, what {prop} did {subj} hold?",
        ],
        "which": [
            "In {year}, which organization was {subj}'s {prop}?",
            "Which organization was the {prop} of {subj} in {year}?",
            "In {year}, which organization was {subj} affiliated with as their {prop}?",
        ],
        "where": [
            "In {year}, where was {subj} located?",
            "Where was {subj} located in {year}?",
            "In {year}, where was {subj} {prop}?",
        ],
    },
    "then": {
        "who": [
            "Who was the {prop} at that time?",
            "Who held the {prop} then?",
        ],
        "what": [
            "What was the {prop} at that time?",
            "What was the {prop} then?",
        ],
        "which": [
            "Which organization was the {prop} at that time?",
            "Which organization was the {prop} then?",
        ],
        "where": [
            "Where was it located at that time?",
            "Where was it located then?",
        ],
    },
    "switch": {
        "who": [
            "In {year2}, who was the {prop}?",
            "By {year2}, who held the {prop}?",
        ],
        "what": [
            "In {year2}, what was the {prop}?",
            "By {year2}, what {prop} did it have?",
        ],
        "which": [
            "In {year2}, which organization was the {prop}?",
            "By {year2}, which organization was the {prop}?",
        ],
        "where": [
            "In {year2}, where was it located?",
            "By {year2}, where was it located?",
        ],
    },
}

PARA_TEMPORAL = {
    "explicit": {
        "who": [
            "Who was the {prop} of {subj} in {year}?",
            "In {year}, who served as the {prop} of {subj}?",
            "Who held the role of {prop} for {subj} in {year}?",
        ],
        "what": [
            "In {year}, what was the {prop} of {subj}?",
            "What {prop} did {subj} have in {year}?",
            "In {year}, what {prop} did {subj} hold?",
        ],
        "which": [
            "In {year}, which organization was {subj}'s {prop}?",
            "Which organization was the {prop} of {subj} in {year}?",
            "In {year}, which organization was {subj} affiliated with as their {prop}?",
        ],
        "where": [
            "In {year}, where was {subj} located?",
            "Where was {subj} located in {year}?",
            "In {year}, where was {subj} {prop}?",
        ],
    }
}


PID_TEMPLATES_EXPLICIT: Dict[str, List[str]] = {
    # leaders
    "P35": [
        "In {year}, who was the head of state of {subj}?",
        "Who served as the head of state of {subj} in {year}?",
        "Who was {subj}'s head of state in {year}?",
    ],
    "P6": [
        "In {year}, who was the head of government of {subj}?",
        "Who served as the head of government of {subj} in {year}?",
        "Who was {subj}'s head of government in {year}?",
    ],
    "P169": [
        "In {year}, who was the CEO of {subj}?",
        "Who served as CEO of {subj} in {year}?",
        "Who was the chief executive officer of {subj} in {year}?",
    ],
    "P488": [
        "In {year}, who was the chairperson of {subj}?",
        "Who served as chairperson of {subj} in {year}?",
        "Who chaired {subj} in {year}?",
    ],
    "P286": [
        "In {year}, who was the head coach of {subj}?",
        "Who served as head coach of {subj} in {year}?",
        "Who coached {subj} in {year}?",
    ],

    # roles and attributes
    "P39": [
        "What position did {subj} hold in {year}?",
        "In {year}, what position did {subj} hold?",
        "Which position did {subj} hold in {year}?",
    ],
    "P102": [
        "In {year}, which political party was {subj} affiliated with?",
        "Which political party was {subj} associated with in {year}?",
        "In {year}, what was {subj}'s political party?",
    ],
    "P106": [
        "What was {subj}'s occupation in {year}?",
        "In {year}, what was {subj}'s occupation?",
        "What occupation did {subj} have in {year}?",
    ],
    "P101": [
        "What was {subj}'s field of work in {year}?",
        "In {year}, what field did {subj} work in?",
        "What field of work was {subj} associated with in {year}?",
    ],
    "P31": [
        "In {year}, what was {subj} an instance of?",
        "What was {subj} an instance of in {year}?",
        "In {year}, what type of thing was {subj}?",
    ],
    "P279": [
        "In {year}, what was {subj} a subclass of?",
        "What was {subj} a subclass of in {year}?",
        "In {year}, {subj} was a subclass of what?",
    ],

    # org relations
    "P108": [
        "In {year}, where did {subj} work?",
        "Where did {subj} work in {year}?",
        "In {year}, which organization employed {subj}?",
    ],
    "P463": [
        "In {year}, which organization was {subj} a member of?",
        "Which organization was {subj} a member of in {year}?",
        "In {year}, {subj} belonged to which organization?",
    ],
    "P69": [
        "In {year}, where did {subj} study?",
        "Where did {subj} study in {year}?",
        "In {year}, which institution did {subj} attend?",
    ],
    "P54": [
        "In {year}, which team did {subj} play for?",
        "Which sports team was {subj} part of in {year}?",
        "In {year}, {subj} played for which team?",
    ],
    "P127": [
        "In {year}, who owned {subj}?",
        "Who owned {subj} in {year}?",
        "In {year}, {subj} was owned by which organization?",
    ],

    # geography
    "P131": [
        "In {year}, where was {subj} located?",
        "Where was {subj} located in {year}?",
        "In {year}, {subj} was located where?",
    ],
    "P276": [
        "In {year}, where was {subj} located?",
        "Where was {subj} located in {year}?",
        "In {year}, what was the location of {subj}?",
    ],
    "P17": [
        "In {year}, which country was {subj} in?",
        "Which country was {subj} located in in {year}?",
        "In {year}, {subj} was in which country?",
    ],
}

P31_PHRASES_TIMELESS = [
    "What type of thing is {subj}?",
    "What kind of thing is {subj}?",
    "How would you categorize {subj}?",
]

P31_PHRASES_TEMPORAL = [
    "In {year}, what type of thing was {subj}?",
    "In {year}, what kind of thing was {subj}?",
    "In {year}, how would you categorize {subj}?",
]


# Generic fallback if PID not in PID_TEMPLATES_EXPLICIT
GENERIC_EXPLICIT = {
    "who": [
        "In {year}, who was the {prop} of {subj}?",
        "Who served as the {prop} of {subj} in {year}?",
    ],
    "what": [
        "In {year}, what was {subj}'s {prop}?",
        "What was the {prop} of {subj} in {year}?",
    ],
    "which": [
        "In {year}, which organization was {subj}'s {prop}?",
        "Which organization was the {prop} of {subj} in {year}?",
    ],
    "where": [
        "In {year}, where was {subj} located?",
        "Where was {subj} located in {year}?",
    ],
}

# Follow-ups that assume subject and/or year are already established in the chain.
FOLLOWUP_PROP_ONLY = {
    "P17":  ["Which country was it in?", "What country was it in?"],
    "P131": ["Where was it located?", "Which administrative area was it located in?"],
    "P276": ["Where was it located?", "What was its location?"],
    "P108": ["Which organization was the employer?", "Where did they work?"],
    "P463": ["Which organization was it a member of?", "What was it a member of?"],
    "P127": ["Who owned it?", "Which organization owned it?"],
    "P169": ["Who was the CEO?", "Who served as CEO?"],
    "P488": ["Who was the chairperson?", "Who served as chairperson?"],
    "P35":  ["Who was the head of state?", "Who served as head of state?"],
    "P6":   ["Who was the head of government?", "Who served as head of government?"],
    "P39":  ["What position was held?", "Which position was held?"],
    "P102": ["Which political party was it?", "What was the political party?"],
    "P31":  ["What was it an instance of?", "What type of thing was it?"],
    "P279": ["What was it a subclass of?", "A subclass of what?"],
    "P54":  ["Which team was it?", "What was the sports team?"],
    "P286": ["Who was the head coach?", "Who coached it?"],
    "P69":  ["Which school did they attend?", "Where were they educated?"],
    "P106": ["What was the occupation?", "What did they do for a living?"],
    "P101": ["What was the field of work?", "Which field did they work in?"],
}

FOLLOWUP_THEN = {
    "P17":  ["Which country was it in then?", "What country was it in at that time?"],
    "P131": ["Where was it located then?", "Where was it located at that time?"],
    "P276": ["Where was it located then?", "What was its location at that time?"],
    "P108": ["Which organization was the employer then?", "Where did they work at that time?"],
    "P54":  ["Which team was it then?", "What was the sports team at that time?"],
}

SCOPE_SWITCH_TEMPLATES = {
    "who": [
        "In {year2}, who was the {prop}?",
        "By {year2}, who was the {prop}?",
    ],
    "what": [
        "In {year2}, what was the {prop}?",
        "By {year2}, what was the {prop}?",
    ],
    "which": [
        "In {year2}, which organization was it associated with for {prop}?",
        "By {year2}, which organization was it associated with for {prop}?",
    ],
    "where": [
        "In {year2}, where was it located?",
        "By {year2}, where was it located?",
    ],
}

# -----------------------------
# Analytic templates (keep small and clean)
# -----------------------------

ANALYTIC_TEMPLATES = {
    "before_after": [
        "Did {event1} happen before {event2}?",
        "Was {event1} earlier than {event2}?",
        "Did {event1} occur after {event2}?",
    ],
    "count_range": [
        "How many times was {entity} associated with {prop} between {year1} and {year2}?",
        "Between {year1} and {year2}, how many {prop} associations did {entity} have?",
    ],
    "comparison": [
        "In {year}, did {entity1} have more {prop} associations than {entity2}?",
        "In {year}, did {entity1} have fewer {prop} associations than {entity2}?",
    ],
}

PARA_TIMELESS = {
    "explicit": {
        "who": [
            "Who was the {prop} of {subj}?",
            "Who served as the {prop} of {subj}?",
            "Who held the role of {prop} for {subj}?",
        ],
        "what": [
            "What was the {prop} of {subj}?",
            "What {prop} did {subj} have?",
            "What {prop} did {subj} hold?",
        ],
        "which": [
            "Which organization was the {prop} of {subj}?",
            "Which organization was {subj}'s {prop}?",
            "Which organization was {subj} affiliated with as their {prop}?",
        ],
        "where": [
            "Where was {subj} located?",
            "Where was {subj} situated?",
        ],
    }
}


# -----------------------------
# Anchor parquet iteration
# -----------------------------

ANCHOR_RE = re.compile(r"truth_(temporal|timeless)_anchor=(\d{4}-\d{2}-\d{2})\.parquet")

def iter_truth_anchors(truth_root: Path, year: int):
    snapshot_dir = truth_root / f"snapshot_year={year}"
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Missing snapshot dir: {snapshot_dir}")
    for p in snapshot_dir.iterdir():
        m = ANCHOR_RE.match(p.name)
        if m:
            yield (m.group(1), m.group(2), p)

# -----------------------------
# Label hygiene
# -----------------------------

_BAD_PREFIXES = ("Q", "P")

def is_clean_label(label: str) -> bool:
    if not label or len(label) < 2:
        return False
    label_norm = " ".join(str(label).strip().lower().split())
    if label_norm in BAD_LABELS:
        return False
    if label_norm.startswith(("unknown ", "none ", "null ")):
        return False
    # Allow names with spaces, hyphens, apostrophes
    return bool(re.match(r"^[A-Za-z0-9\s\-'\.]{2,100}$", label))


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def is_placeholder_answer(answer: Optional[str]) -> bool:
    if not answer:
        return True
    ans = normalize_text(answer).lower()
    if ans in BAD_LABELS:
        return True
    if ans.startswith(("unknown", "none", "null", "n/a")):
        return True
    return False


def qa_is_high_quality(question: str, answer: str) -> bool:
    q = normalize_text(question)
    a = normalize_text(answer)
    if not q or not a:
        return False
    if is_placeholder_answer(a):
        return False
    if len(q) < 10 or len(q) > 180:
        return False
    if len(a) < 2 or len(a) > 120:
        return False
    if not q.endswith("?"):
        return False
    return True


def chain_is_high_quality(turns: List[Dict[str, Any]]) -> bool:
    if not turns:
        return False
    q_seen = set()
    a_seen = []
    for t in turns:
        q = normalize_text(t.get("question", ""))
        a = normalize_text(t.get("answer", ""))
        if not qa_is_high_quality(q, a):
            return False
        q_norm = q.lower()
        if q_norm in q_seen:
            return False
        q_seen.add(q_norm)
        a_seen.append(a.lower())
    # Avoid chains that loop over the exact same answer repeatedly.
    if len(a_seen) >= 3 and len(set(a_seen)) == 1:
        return False
    return True


def turn_from_row(question: str, row: Dict[str, Any], year: Optional[int] = None) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": row["value_label"],
        "year": row.get("year") if year is None else year,
        "pid": row.get("pid"),
        "subject_label": row.get("subject_label"),
        "subject_qid": row.get("subject_qid"),
        "value_label": row.get("value_label"),
        "value_qid": row.get("value_qid"),
    }


def followup_reintroduce_subject(pid: str, subj: str, truth_type: str, mention_time: bool = False) -> str:
    forms = PID_REALIZATION.get(pid, {})
    if mention_time and truth_type != "timeless":
        opts = forms.get("followup_then")
        if opts:
            base = pick(opts).rstrip("?")
            return f"And for {subj}, {base.lower()}?"
    opts = forms.get("followup")
    if opts:
        base = pick(opts).rstrip("?")
        return f"And for {subj}, {base.lower()}?"
    return explicit_question(pid, subj, 0, "timeless")


from typing import Optional

def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

from typing import Optional, Dict, Any

def extract_year_from_qualifiers(row: Dict[str, Any]) -> Optional[int]:
    """
    Extract a year from stage2 truth temporal rows.

    Supported columns (current schema):
      - pit_date
      - start_date
      - end_date

    Backward-compatible fallbacks:
      - q_point_in_time
      - q_start_time
      - q_end_time
    """
    for key in (
        "pit_date",
        "start_date",
        "end_date",
        "q_point_in_time",
        "q_start_time",
        "q_end_time",
    ):
        val = row.get(key)
        if not val:
            continue
        if isinstance(val, list) and val:
            val = val[0]
        try:
            return int(str(val)[:4])
        except Exception:
            continue
    return None




def pick(xs):
    return random.choice(xs)

def explicit_question(pid: str, subj: str, year: int, truth_type: str) -> str:
    forms = PID_REALIZATION.get(pid)
    if forms:
        key = "explicit_timeless" if truth_type == "timeless" else "explicit_temporal"
        if key in forms:
            return pick(forms[key]).format(subj=subj, year=year)

    # Special-case fallback: avoid "instance of" phrasing
    if pid == "P31":
        if truth_type == "timeless":
            return pick(P31_PHRASES_TIMELESS).format(subj=subj)
        return pick(P31_PHRASES_TEMPORAL).format(subj=subj, year=year)

    spec = PID_SPECS[pid]
    if truth_type == "timeless":
        return pick(PARA_TIMELESS["explicit"][spec["qtype"]]).format(prop=spec["surface"], subj=subj)
    return pick(PARA_TEMPORAL["explicit"][spec["qtype"]]).format(prop=spec["surface"], subj=subj, year=year)


def followup_switch_year(pid: str, year2: int) -> str:
    forms = PID_REALIZATION.get(pid)
    if forms and "switch" in forms:
        return pick(forms["switch"]).format(year2=year2)

    spec = PID_SPECS[pid]
    prop = spec["surface"]
    qtype = spec["qtype"]

    if qtype == "who":
        return pick([
            "In {year2}, who was the {prop}?",
            "By {year2}, who held the {prop}?",
        ]).format(year2=year2, prop=prop)

    if qtype == "what":
        if pid == "P31":
            return pick([
                "In {year2}, what type of thing was it?",
                "By {year2}, what kind of thing was it?",
            ]).format(year2=year2)
        return pick([
            "In {year2}, what was the {prop}?",
            "By {year2}, what {prop} did it have?",
        ]).format(year2=year2, prop=prop)

    if qtype == "which":
        return pick([
            "In {year2}, which organization was the {prop}?",
            "By {year2}, which organization was the {prop}?",
        ]).format(year2=year2, prop=prop)

    if qtype == "where":
        return pick([
            "In {year2}, where was it located?",
            "By {year2}, where was it located?",
        ]).format(year2=year2)

    return f"In {year2}, what was the {prop}?"



def followup_prop_only(pid: str) -> str:
    forms = PID_REALIZATION.get(pid)
    if forms and "followup" in forms:
        return pick(forms["followup"])
    if pid in FOLLOWUP_PROP_ONLY:
        return pick(FOLLOWUP_PROP_ONLY[pid])
    spec = PID_SPECS[pid]
    if spec["qtype"] == "who":
        return f"Who was the {spec['surface']}?"
    if spec["qtype"] == "where":
        return "Where was it located?"
    if spec["qtype"] == "what":
        return f"What was the {spec['surface']}?"
    return f"Which organization was the {spec['surface']}?"

def followup_then(pid: str) -> str:
    forms = PID_REALIZATION.get(pid)
    if forms and "followup_then" in forms:
        return pick(forms["followup_then"])
    if pid in FOLLOWUP_THEN:
        return pick(FOLLOWUP_THEN[pid])
    q = followup_prop_only(pid)
    return q[:-1] + " at that time?" if q.endswith("?") else (q + " at that time?")

def followup_scope_switch(pid: str, year2: int) -> str:
    spec = PID_SPECS[pid]
    return pick(SCOPE_SWITCH_TEMPLATES[spec["qtype"]]).format(prop=spec["surface"], year2=year2)

# -----------------------------
# Builders
# Each returns either:
#   - single QA dict (self-contained)
#   - list of turn dicts (for chains)
# Turn dict schema: {"question":..., "answer":..., "year":..., "pid":..., "subject_label":..., "value_label":...}
# -----------------------------

def build_direct(row: dict, truth_type: str) -> dict:
    pid = row["pid"]
    subj = row["subject_label"]
    year = row["year"]
    q = explicit_question(pid, subj, year, truth_type)
    return {
        "question": q,
        "answer": row["value_label"],
        "pid": pid,
        "subject_label": subj,
        "subject_qid": row.get("subject_qid"),
        "value_label": row["value_label"],
        "value_qid": row.get("value_qid"),
        "family": "direct",
    }
def switch_question(pid: str, year2: int) -> str:
    spec = PID_SPECS[pid]
    qtype = spec["qtype"]
    prop = spec["surface"]

    if qtype == "who":
        return pick([
            "In {year2}, who was the {prop}?",
            "By {year2}, who held the {prop}?",
        ]).format(year2=year2, prop=prop)

    if qtype == "what":
        # Special-case P31 to avoid “instance of”
        if pid == "P31":
            return pick([
                "In {year2}, what type of thing was it?",
                "By {year2}, what kind of thing was it?",
            ]).format(year2=year2)
        return pick([
            "In {year2}, what was the {prop}?",
            "By {year2}, what {prop} did it have?",
        ]).format(year2=year2, prop=prop)

    if qtype == "which":
        return pick([
            "In {year2}, which organization was the {prop}?",
            "By {year2}, which organization was the {prop}?",
        ]).format(year2=year2, prop=prop)

    if qtype == "where":
        return pick([
            "In {year2}, where was it located?",
            "By {year2}, where was it located?",
        ]).format(year2=year2)

    return f"In {year2}, what was the {prop}?"


def build_carryover(row: dict, truth_type: str) -> list[dict]:
    pid1 = row["pid"]
    pid2 = COMPLEMENTS.get(pid1, pid1)
    subj = row["subject_label"]
    year = row["year"]

    q1 = explicit_question(pid1, subj, year, truth_type)

    # Followup (prop only or "then")
    q2 = followup_then(pid2) if (truth_type != "timeless") else followup_prop_only(pid2)

    return [
        {"question": q1, "answer": row["value_label"], "family": "carryover"},
        {"question": q2, "answer": row["value_label"], "family": "carryover_then" if "then" in q2.lower() else "carryover"},
    ]


def build_carryover_then(row: dict) -> List[dict]:
    # Same as carryover but second turn uses "at that time/then".
    return build_carryover(row)  # second turn style chosen later

def build_cross_entity_then(r1: Dict[str, Any], r2: Dict[str, Any], truth_type: str) -> List[Dict[str, Any]]:
    """
    Same pid, different subjects. First turn sets year, second uses then.
    """
    pid = r1["pid"]
    year = r1["year"]
    t1 = turn_from_row(explicit_question(pid, r1["subject_label"], year, truth_type), r1, year=year)

    if truth_type == "timeless":
        q2 = f"And for {r2['subject_label']}, {followup_prop_only(pid).rstrip('?')}?"
    else:
        q2 = f"And for {r2['subject_label']}, {followup_then(pid)}"

    t2 = turn_from_row(q2, r2, year=year)
    return [t1, t2]

def build_scope_switch(r1: Dict[str, Any], r2: Dict[str, Any], truth_type: str) -> Optional[List[Dict[str, Any]]]:
    """
    Same subject+pid, different years (from qualifiers).
    """
    if truth_type == "timeless":
        return None

    pid = r1["pid"]
    subj = r1["subject_label"]
    y1 = r1["year"]
    y2 = r2["year"]
    if y1 == y2:
        return None

    t1 = turn_from_row(explicit_question(pid, subj, y1, truth_type), r1, year=y1)
    # second turn explicitly switches year, but keeps entity implicit to be conversational
    spec = PID_SPECS[pid]
    q2 = switch_question(pid, y2)
    t2 = turn_from_row(q2, r2, year=y2)
    return [t1, t2]


def build_long_chain(subject: str, year: int, rows_for_subject: List[dict], min_len=3, max_len=7) -> Optional[List[dict]]:
    # Build a coherent chain for one subject at one year: multiple different properties.
    # Turn 1 explicit, remaining turns implicit "then" follow-ups.
    rows = [r for r in rows_for_subject if r.get("year") == year and r.get("pid") in PID_SPECS]
    if len(rows) < min_len:
        return None

    # Prefer diverse pids
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["pid"], r)
    pid_rows = list(by_pid.values())
    if len(pid_rows) < min_len:
        return None

    random.shuffle(pid_rows)
    L = random.randint(min_len, min(max_len, len(pid_rows)))
    chosen = pid_rows[:L]

    turns = []
    used_q = set()
    for i, r in enumerate(chosen):
        pid = r["pid"]
        if i == 0:
            q = explicit_question(pid, subject, year)
        else:
            # keep subject implicit but property explicit
            q = followup_then(pid)
        if q.lower().strip() in used_q:
            continue
        used_q.add(q.lower().strip())
        turns.append({
            "question": q,
            "answer": r["value_label"],
            "year": year,
            "pid": pid,
            "subject_label": subject,
            "value_label": r["value_label"],
        })

    if len(turns) < min_len:
        return None
    return turns

import random
from collections import defaultdict

def pick(xs: List[str]) -> str:
    return random.choice(xs)

def unique_clean(values):
    out = []
    seen = set()
    for v in values:
        if not v:
            continue
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def group_by_subject_pid(rows):
    g = defaultdict(list)
    for r in rows:
        subj = r.get("subject_label")
        pid = r.get("pid")
        if subj and pid:
            g[(subj, pid)].append(r)
    return g

def group_by_subject(rows):
    g = defaultdict(list)
    for r in rows:
        subj = r.get("subject_label")
        if subj:
            g[subj].append(r)
    return g

MULTIHOP2_TEMPLATES = {
    "employer_country": [
        "In {year}, {subj} worked at {mid}. Which country was {mid} in that year?",
        "In {year}, {subj} worked at {mid}. In {year}, which country was {mid} located in?",
        "{subj} worked at {mid} in {year}. What country was {mid} in that year?",
    ],
    "team_country": [
        "In {year}, {subj} played for {mid}. Which country was {mid} in that year?",
        "{subj} played for {mid} in {year}. In {year}, which country was the team based in?",
    ],
    "admin_country": [
        "In {year}, {subj} was located in {mid}. Which country was {mid} in that year?",
        "{subj} was located in {mid} in {year}. In {year}, what country was {mid} in?",
    ],
    "ownedby_ceo": [
        "In {year}, {subj} was owned by {mid}. Who was the CEO of {mid} in {year}?",
        "{subj} was owned by {mid} in {year}. In {year}, who served as CEO of {mid}?",
    ],
    "instance_superclass": [
        "In {year}, {subj} was an instance of {mid}. What was {mid} a subclass of in {year}?",
        "{subj} was an instance of {mid} in {year}. In {year}, {mid} was a subclass of what?",
    ],
}

def build_multihop_2(r1: Dict[str, Any], r2: Dict[str, Any], truth_type: str) -> Dict[str, Any]:
    """
    r1: subj -> mid via pid1
    r2: mid -> val2 via pid2
    Example: Person worked at Org. Org is in Country. Which country did person work in?
    """
    subj = r1["subject_label"]
    mid = r1["value_label"]
    pid2 = r2["pid"]
    year = r1["year"]

    # Ask about the second hop in terms of the original subject
    if pid2 == "P17":
        q = f"In {year}, which country was {subj} associated with through {mid}?"
    else:
        q = f"In {year}, what was {subj}'s {PID_SPECS[pid2]['surface']} via {mid}?"

    if truth_type == "timeless":
        if pid2 == "P17":
            q = f"Which country was {subj} associated with through {mid}?"
        else:
            q = f"What was {subj}'s {PID_SPECS[pid2]['surface']} via {mid}?"

    return {
        "question": q,
        "answer": r2["value_label"],
        "pid": pid2,
        "subject_label": mid,
        "subject_qid": r1.get("value_qid"),
        "value_label": r2["value_label"],
        "value_qid": r2.get("value_qid"),
        "family": "multihop_2",
    }


def build_multihop_3(r1: Dict[str, Any], r2: Dict[str, Any], r3: Dict[str, Any], truth_type: str) -> Dict[str, Any]:
    """
    Pattern: P108 (person->org), P127 (org->parent), P169 (parent->CEO)
    """
    year = r1["year"]
    person = r1["subject_label"]
    org = r1["value_label"]
    parent = r2["value_label"]
    ceo = r3["value_label"]

    if truth_type == "timeless":
        q = (
            f"{person} worked at {org}. {org} was owned by {parent}. "
            f"Who was the {PID_SPECS[r3['pid']]['surface']} of {parent}?"
        )
    else:
        q = (
            f"In {year}, {person} worked at {org}. {org} was owned by {parent}. "
            f"Who was the {PID_SPECS[r3['pid']]['surface']} of {parent} at that time?"
        )

    return {
        "question": q,
        "answer": ceo,
        "pid": r3.get("pid"),
        "subject_label": parent,
        "subject_qid": r2.get("value_qid"),
        "value_label": ceo,
        "value_qid": r3.get("value_qid"),
        "family": "multihop_3",
    }


# -----------------------------
# Analytic builders (single-turn, self-contained)
# -----------------------------

def build_before_after(row1: dict, row2: dict) -> Optional[dict]:
    y1 = extract_year_from_qualifiers(row1)
    y2 = extract_year_from_qualifiers(row2)
    if not y1 or not y2:
        return None

    event1 = f"{row1['subject_label']} ({row1['value_label']})"
    event2 = f"{row2['subject_label']} ({row2['value_label']})"

    q = pick(ANALYTIC_TEMPLATES["before_after"]).format(event1=event1, event2=event2)

    # choose polarity based on template
    ql = q.lower()
    if "after" in ql:
        ans = "Yes" if y1 > y2 else "No"
    else:
        ans = "Yes" if y1 < y2 else "No"

    return {
        "question": q,
        "answer": ans,
        "family": "analytic_before_after",
        "year": row1.get("year"),
    }

def build_count_range(rows: List[dict], entity: str, pid: str, y1: int, y2: int) -> dict:
    cnt = 0
    for r in rows:
        ry = extract_year_from_qualifiers(r)
        if ry and y1 <= ry <= y2:
            cnt += 1

    q = pick(ANALYTIC_TEMPLATES["count_range"]).format(
        entity=entity,
        prop=PID_SPECS[pid]["surface"],
        year1=y1,
        year2=y2
    )
    return {
        "question": q,
        "answer": str(cnt),
        "family": "analytic_count_range",
        "year": rows[0].get("year"),
    }

# -----------------------------
# Difficulty helpers
# -----------------------------

def difficulty_for_chain(turns_len: int, analytic: bool) -> Dict:
    return {"hops": turns_len, "temporal": True, "analytic": analytic}

def difficulty_for_single(family: str) -> Dict:
    analytic = family.startswith("analytic_")
    if family == "direct":
        hops = 1
    elif family == "multihop_2":
        hops = 2
    elif family == "multihop_3":
        hops = 3
    else:
        hops = 1
    return {"hops": hops, "temporal": True, "analytic": analytic}


ARITH_TEMPLATES = {
    "count_distinct_year": [
        "In {year}, how many distinct {prop_plural} did {subj} have?",
        "How many different {prop_plural} was {subj} associated with in {year}?",
    ],
    "diff_count_two_years": [
        "How many more distinct {prop_plural} did {subj} have in {year2} than in {year1}?",
        "In {year2} compared with {year1}, by how many did the number of distinct {prop_plural} for {subj} change?",
    ],
    "sum_two_entities": [
        "In {year}, what is the total number of distinct {prop_plural} for {e1} and {e2} combined?",
        "In {year}, how many distinct {prop_plural} did {e1} and {e2} have in total?",
    ],
    "diff_two_entities": [
        "In {year}, how many more distinct {prop_plural} did {e1} have than {e2}?",
        "In {year}, what was the difference in the number of distinct {prop_plural} between {e1} and {e2}?",
    ],
}

# minimal plural map for nice English
PROP_PLURALS = {
    "P108": "employers",
    "P54": "teams",
    "P463": "memberships",
    "P39": "positions",
    "P69": "schools",
    "P131": "locations",
    "P17": "countries",
    "P31": "types",
}

def prop_plural(surface: str) -> str:
    # simple pluralization; good enough for benchmark text
    if surface.endswith("y") and not surface.endswith(("ay", "ey", "iy", "oy", "uy")):
        return surface[:-1] + "ies"
    if surface.endswith("s"):
        return surface
    return surface + "s"

def _distinct_values_in_year(rows: List[Dict[str, Any]], pid: str, year: int) -> Set[str]:
    vals: Set[str] = set()
    for r in rows:
        if r.get("pid") == pid and r.get("year") == year and is_clean_label(r.get("value_label")):
            vals.add(r["value_label"])
    return vals


def build_count_distinct_in_year(rows: List[Dict[str, Any]], pid: str, year: int, truth_type: str) -> Optional[Dict[str, Any]]:
    vals = _distinct_values_in_year(rows, pid, year)
    if len(vals) < 2:
        return None

    subj = rows[0]["subject_label"]
    prop = PID_SPECS[pid]["surface"]
    prop_p = prop_plural(prop)

    if truth_type == "timeless":
        q = f"How many distinct {prop_p} is {subj} associated with?"
    else:
        q = f"In {year}, how many distinct {prop_p} was {subj} associated with?"

    return {"question": q, "answer": str(len(vals)), "family": "complex_count_distinct"}

def _distinct_values_in_year(rows: List[Dict[str, Any]], pid: str, year: int) -> Set[str]:
    vals: Set[str] = set()
    for r in rows:
        if r.get("pid") == pid and r.get("year") == year and is_clean_label(r.get("value_label")):
            vals.add(r["value_label"])
    return vals


def build_count_distinct_in_year(rows: List[Dict[str, Any]], pid: str, year: int, truth_type: str) -> Optional[Dict[str, Any]]:
    vals = _distinct_values_in_year(rows, pid, year)
    if len(vals) < 2:
        return None

    subj = rows[0]["subject_label"]
    prop = PID_SPECS[pid]["surface"]
    prop_p = prop_plural(prop)

    if truth_type == "timeless":
        q = f"How many distinct {prop_p} is {subj} associated with?"
    else:
        q = f"In {year}, how many distinct {prop_p} was {subj} associated with?"

    return {
        "question": q,
        "answer": str(len(vals)),
        "pid": pid,
        "subject_label": subj,
        "subject_qid": rows[0].get("subject_qid"),
        "family": "complex_count_distinct",
    }


import polars as pl
from typing import Dict

def cap_df(
    df: pl.DataFrame,
    truth_type: str,
    max_rows_temporal: int,
    max_rows_timeless: int,
    seed: int,
) -> pl.DataFrame:
    cap = max_rows_temporal if truth_type == "temporal" else max_rows_timeless
    if cap is None or cap <= 0 or df.height <= cap:
        return df
    return df.sample(n=cap, with_replacement=False, seed=seed)


def collect_qids(df: pl.DataFrame) -> Set[str]:
    qids: Set[str] = set()
    if "subject_qid" in df.columns:
        for x in df.get_column("subject_qid").drop_nulls().to_list():
            if isinstance(x, str) and x and x[0] in ("Q", "P"):
                qids.add(x)
    if "value_qid" in df.columns:
        for x in df.get_column("value_qid").drop_nulls().to_list():
            if isinstance(x, str) and x and x[0] in ("Q", "P"):
                qids.add(x)
    return qids


def load_en_labels_from_parquet(labels_parquet: Path, qids: Iterable[str]) -> Dict[str, str]:
    """
    Reads English labels from your qid_labels_desc.parquet.
    Expected columns: id, label, and a language col 'lang' or 'len'.
    """
    qids = list(set(qids))
    if not qids:
        return {}

    schema = pl.scan_parquet(str(labels_parquet)).collect_schema()
    lang_col = "lang" if "lang" in schema else ("len" if "len" in schema else None)

    cols = ["id", "label"] + ([lang_col] if lang_col else [])
    lf = pl.scan_parquet(str(labels_parquet)).select(cols)
    if lang_col:
        lf = lf.filter(pl.col(lang_col) == "en")

    lf = lf.filter(pl.col("id").is_in(qids)).select(["id", "label"]).drop_nulls()
    df = lf.collect()
    return dict(zip(df["id"].to_list(), df["label"].to_list()))


def attach_labels(df: pl.DataFrame, label_map: Dict[str, str]) -> pl.DataFrame:
    if "subject_label" not in df.columns and "subject_qid" in df.columns:
        df = df.with_columns(
            pl.col("subject_qid")
              .map_elements(lambda x: label_map.get(x, None), return_dtype=pl.Utf8)
              .alias("subject_label")
        )
    if "value_label" not in df.columns and "value_qid" in df.columns:
        df = df.with_columns(
            pl.col("value_qid")
              .map_elements(lambda x: label_map.get(x, None), return_dtype=pl.Utf8)
              .alias("value_label")
        )
    return df


def drop_unlabeled(df: pl.DataFrame) -> pl.DataFrame:
    if "subject_label" in df.columns:
        df = df.filter(pl.col("subject_label").is_not_null())
    if "value_label" in df.columns:
        df = df.filter(pl.col("value_label").is_not_null())
    return df


def build_multi_turn_chain_for_subject(
    subj: str,
    year: int,
    truth_type: str,
    by_subject_pid: dict,
    min_len: int,
    max_len: int,
    then_prob: float = 0.35,
):
    """
    Build a multi-turn chain for any subject with >= min_len clean properties.
    No rigid plans — just uses whatever is available.
    """
    # Get all valid PIDs for this subject
    pids = []
    for (s, pid), rows in by_subject_pid.items():
        if s == subj and pid in PID_SPECS:
            # Pick one clean row
            clean_rows = [r for r in rows if is_clean_label(r.get("value_label"))]
            if clean_rows:
                pids.append(pid)
    
    if len(pids) < min_len:
        return None

    # Prefer family diversity so long chains feel less synthetic.
    random.shuffle(pids)
    chosen_pids = []
    used_families = set()
    target_len = random.randint(min_len, min(max_len, len(pids)))
    for pid in pids:
        fam = RELATION_FAMILIES.get(pid, pid)
        if fam not in used_families or len(chosen_pids) < min_len:
            chosen_pids.append(pid)
            used_families.add(fam)
        if len(chosen_pids) >= target_len:
            break
    if len(chosen_pids) < min_len:
        chosen_pids = pids[:target_len]

    turns = []
    used_answers = set()  # avoid duplicate answers
    used_questions = set()

    for i, pid in enumerate(chosen_pids):
        # Pick a clean row
        clean_rows = [r for r in by_subject_pid[(subj, pid)] if is_clean_label(r.get("value_label"))]
        if not clean_rows:
            continue
        r = random.choice(clean_rows)
        
        # Skip if answer already used
        if r["value_label"] in used_answers:
            continue
        used_answers.add(r["value_label"])

        if i == 0:
            q = explicit_question(pid, subj, year, truth_type)
        else:
            if i >= 3 and random.random() < 0.4:
                q = followup_reintroduce_subject(pid, subj, truth_type, mention_time=(truth_type == "temporal"))
            elif truth_type == "temporal" and random.random() < then_prob:
                q = followup_then(pid)
            else:
                q = followup_prop_only(pid)

        # Dedup questions
        q_norm = " ".join(q.lower().split())
        if q_norm in used_questions:
            continue
        used_questions.add(q_norm)

        turns.append({
            "question": q,
            "answer": r["value_label"],
            "year": year,
            "pid": pid,
            "subject_label": subj,
            "value_label": r["value_label"],
        })

        if len(turns) >= max_len:
            break

    return turns if len(turns) >= min_len else None


def build_temporal_narrative_chain(subj: str, pid: str, rows: List[Dict[str, Any]], min_len: int = 3, max_len: int = 6) -> Optional[List[Dict[str, Any]]]:
    by_year: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        y = extract_year_from_qualifiers(r)
        if y is not None and is_clean_label(r.get("value_label")):
            by_year[y].append(r)
    years = sorted(by_year.keys())
    if len(years) < min_len:
        return None

    sample_len = min(max_len, len(years))
    if sample_len > min_len:
        sample_len = random.randint(min_len, sample_len)
    chosen_years = years[:sample_len] if len(years) == sample_len else sorted(random.sample(years, sample_len))

    turns: List[Dict[str, Any]] = []
    for idx, y in enumerate(chosen_years):
        row = random.choice(by_year[y])
        if idx == 0:
            q = explicit_question(pid, subj, y, "temporal")
        elif idx == 1:
            q = followup_switch_year(pid, y)
        elif idx == len(chosen_years) - 1 and random.random() < 0.5:
            q = f"And by {y}, {followup_prop_only(pid).rstrip('?').lower()}?"
        else:
            q = f"What about in {y}?"
        turns.append(turn_from_row(q, row, year=y))
    return turns if chain_is_high_quality(turns) else None


def build_change_point_chain(
    subj: str,
    pid_main: str,
    pid_follow: str,
    rows_main: List[Dict[str, Any]],
    rows_follow: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    by_year_main: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_year_follow: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_main:
        y = extract_year_from_qualifiers(r)
        if y is not None and is_clean_label(r.get("value_label")):
            by_year_main[y].append(r)
    for r in rows_follow:
        y = extract_year_from_qualifiers(r)
        if y is not None and is_clean_label(r.get("value_label")):
            by_year_follow[y].append(r)

    common_years = sorted(set(by_year_main.keys()) & set(by_year_follow.keys()))
    if len(common_years) < 2:
        return None
    y1, y2 = random.sample(common_years, 2)
    r1 = random.choice(by_year_main[y1])
    r1f = random.choice(by_year_follow[y1])
    r2 = random.choice(by_year_main[y2])
    r2f = random.choice(by_year_follow[y2])

    turns = [
        turn_from_row(explicit_question(pid_main, subj, y1, "temporal"), r1, year=y1),
        turn_from_row(followup_then(pid_follow), r1f, year=y1),
        turn_from_row(followup_switch_year(pid_main, y2), r2, year=y2),
        turn_from_row(followup_then(pid_follow), r2f, year=y2),
    ]
    return turns if chain_is_high_quality(turns) else None


ANCHOR_RE = re.compile(r"truth_(temporal|timeless)_anchor=(\d{4}-\d{2}-\d{2})\.parquet")

# For a carryover chain: [Q1: Who was CEO?, Q2: Who owned it?]
# Extend with:
#   Q3: Where is the owner headquartered? → P159 or P131
#   Q4: Who is the current CEO of the owner? → P169

def extend_carryover_chain(carryover_turns, by_subject, truth_type, year):
    last_answer = carryover_turns[-1]["answer"]
    if last_answer not in by_subject:
        return carryover_turns
    ext_rows = by_subject[last_answer]
    for r in random.sample(ext_rows, min(3, len(ext_rows))):
        pid = r["pid"]
        if pid in PID_SPECS:
            q = followup_then(pid) if truth_type == "temporal" else followup_prop_only(pid)
            carryover_turns.append({
                "question": q,
                "answer": r["value_label"],
                "year": year,
                "pid": pid,
                "subject_label": last_answer,
                "value_label": r["value_label"],
            })
            if len(carryover_turns) >= 5:
                break
    return carryover_turns
    
def iter_truth_anchors(truth_root: Path, year: int):
    snapshot_dir = truth_root / f"snapshot_year={year}"
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Missing snapshot dir: {snapshot_dir}")
    for p in snapshot_dir.iterdir():
        m = ANCHOR_RE.match(p.name)
        if m:
            yield (m.group(1), m.group(2), p)

            
def load_anchor_rows(truth_path: Path, snapshot_year: int, truth_type: str, label_parquet: Path,
                     max_rows_temporal: int, max_rows_timeless: int, seed: int) -> List[Dict]:
    df = pl.read_parquet(truth_path)
    df = cap_df(df, truth_type, max_rows_temporal, max_rows_timeless, seed)

    need_labels = ("subject_label" not in df.columns) or ("value_label" not in df.columns)
    if need_labels:
        qids = collect_qids(df)
        label_map = load_en_labels_from_parquet(label_parquet, qids)
        df = attach_labels(df, label_map)
        df = drop_unlabeled(df)

    if "subject_label" in df.columns:
        df = df.filter(pl.col("subject_label").map_elements(is_clean_label, return_dtype=pl.Boolean))
    if "value_label" in df.columns:
        df = df.filter(pl.col("value_label").map_elements(is_clean_label, return_dtype=pl.Boolean))

    rows = df.to_dicts()
    for r in rows:
        r["year"] = snapshot_year
    return rows


def all_active_truth_path(truth_path: Path) -> Path:
    name = truth_path.name
    if "_all_active_" in name:
        return truth_path
    if "truth_temporal_anchor=" in name:
        return truth_path.with_name(name.replace("truth_temporal_anchor=", "truth_temporal_all_active_anchor="))
    if "truth_timeless_anchor=" in name:
        return truth_path.with_name(name.replace("truth_timeless_anchor=", "truth_timeless_all_active_anchor="))
    return truth_path


def resolve_present_day_truth_path(
    truth_root: Path,
    truth_type: str,
    present_day_snapshot_year: int,
    present_day_anchor_date: Optional[str],
) -> Optional[Path]:
    anchors = sorted(iter_truth_anchors(truth_root, present_day_snapshot_year), key=lambda x: x[1])
    anchors = [x for x in anchors if x[0] == truth_type]
    if present_day_anchor_date is not None:
        anchors = [x for x in anchors if x[1] == present_day_anchor_date]
    if not anchors:
        return None
    return anchors[-1][2]


def cap_enabled(cap: Optional[int]) -> bool:
    return cap is not None and cap > 0


def reached_cap(count: int, cap: Optional[int]) -> bool:
    return cap_enabled(cap) and count >= cap


def build_present_day_lookup(
    truth_root: Path,
    label_parquet: Path,
    truth_type: str,
    present_day_snapshot_year: int,
    present_day_anchor_date: Optional[str],
    max_rows_temporal: int,
    max_rows_timeless: int,
    seed: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    truth_path = resolve_present_day_truth_path(
        truth_root=truth_root,
        truth_type=truth_type,
        present_day_snapshot_year=present_day_snapshot_year,
        present_day_anchor_date=present_day_anchor_date,
    )
    if truth_path is None:
        return {}
    rows = load_anchor_rows(
        truth_path=truth_path,
        snapshot_year=present_day_snapshot_year,
        truth_type=truth_type,
        label_parquet=label_parquet,
        max_rows_temporal=max_rows_temporal,
        max_rows_timeless=max_rows_timeless,
        seed=seed,
    )
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        subj = r.get("subject_label")
        pid = r.get("pid")
        val = r.get("value_label")
        if pid in PID_SPECS and is_clean_label(subj) and is_clean_label(val):
            lookup[(subj, pid)] = r
    return lookup

def build_cross_anchor_chain(
    subj: str,
    plan: List[str],
    year_a: int,
    year_b: int,
    truth_type: str,
    by_subject_pid_a: Dict[Tuple[str,str], List[Dict]],
    by_subject_pid_b: Dict[Tuple[str,str], List[Dict]],
    then_prob: float = 0.35,
) -> Optional[List[Dict]]:
    # Require at least 2 PIDs in the plan
    if len(plan) < 2:
        return None
        
    # Require all PIDs in plan to exist in both anchors
    for pid in plan:
        if (subj, pid) not in by_subject_pid_a or (subj, pid) not in by_subject_pid_b:
            return None

    turns: List[Dict] = []
    # Turn 1: explicit from anchor A
    pid0 = plan[0]
    r0 = random.choice(by_subject_pid_a[(subj, pid0)])
    turns.append(turn_from_row(explicit_question(pid0, subj, year_a, truth_type), r0, year=year_a))
    
    # Turn 2: follow-up from anchor A (if plan has ≥2)
    if len(plan) >= 2:
        pid1 = plan[1]
        r1 = random.choice(by_subject_pid_a[(subj, pid1)])
        q1 = followup_then(pid1) if (truth_type != "timeless" and random.random() < then_prob) else followup_prop_only(pid1)
        turns.append(turn_from_row(q1, r1, year=year_a))
    
    # Turn 3: switch to anchor B (explicit year switch)
    r0b = random.choice(by_subject_pid_b[(subj, pid0)])
    q_switch = followup_switch_year(pid0, year_b)  # or switch_question
    turns.append(turn_from_row(q_switch, r0b, year=year_b))
    
    # Turns 4+: continue from anchor B
    for pid in plan[2:]:
        rb = random.choice(by_subject_pid_b[(subj, pid)])
        q = followup_then(pid) if random.random() < then_prob else followup_prop_only(pid)
        turns.append(turn_from_row(q, rb, year=year_b))
    
    return turns if len(turns) >= 3 else None


def rows_for_subject_pid_in_year(subject_rows: list[dict], pid: str, year: int, truth_type: str) -> list[dict]:
    if truth_type == "timeless":
        return [r for r in subject_rows if r.get("pid") == pid]
    out = []
    for r in subject_rows:
        if r.get("pid") != pid:
            continue
        y = extract_year_from_qualifiers(r)
        if y == year:
            out.append(r)
    return out

def build_sum_two_entities(rows_e1: list[dict], rows_e2: list[dict], pid: str, year: int, truth_type: str):
    a = rows_for_subject_pid_in_year(rows_e1, pid, year, truth_type)
    b = rows_for_subject_pid_in_year(rows_e2, pid, year, truth_type)
    if not a or not b:
        return None
    # count distinct values
    ca = len(set(r.get("value_label") for r in a if r.get("value_label")))
    cb = len(set(r.get("value_label") for r in b if r.get("value_label")))
    if ca == 0 or cb == 0:
        return None

    e1 = a[0].get("subject_label")
    e2 = b[0].get("subject_label")
    prop = PID_SPECS[pid]["surface"]

    q = pick([
        "In {year}, what is the total number of distinct {prop} across {e1} and {e2}?",
        "In {year}, how many distinct {prop} did {e1} and {e2} have in total?",
        "Add them up: in {year}, total distinct {prop} for {e1} plus {e2}?",
    ]).format(year=year, prop=prop, e1=e1, e2=e2)

    return {
        "question": q,
        "answer": str(ca + cb),
        "pid": pid,
        "subject_label": e1,
        "subject_qid": a[0].get("subject_qid"),
        "family": "complex_sum_two_entities",
    }

def build_diff_two_entities(rows_e1: list[dict], rows_e2: list[dict], pid: str, year: int, truth_type: str):
    a = rows_for_subject_pid_in_year(rows_e1, pid, year, truth_type)
    b = rows_for_subject_pid_in_year(rows_e2, pid, year, truth_type)
    if not a or not b:
        return None

    ca = len(set(r.get("value_label") for r in a if r.get("value_label")))
    cb = len(set(r.get("value_label") for r in b if r.get("value_label")))
    if ca == 0 or cb == 0 or ca == cb:
        return None

    e1 = a[0].get("subject_label")
    e2 = b[0].get("subject_label")
    prop = PID_SPECS[pid]["surface"]

    q = pick([
        "In {year}, what is the difference in the number of distinct {prop} between {e1} and {e2}?",
        "In {year}, how many more distinct {prop} did {e1} have than {e2}?",
        "Compute the difference: in {year}, distinct {prop} of {e1} minus {e2}?",
    ]).format(year=year, prop=prop, e1=e1, e2=e2)

    return {
        "question": q,
        "answer": str(ca - cb),
        "pid": pid,
        "subject_label": e1,
        "subject_qid": a[0].get("subject_qid"),
        "family": "complex_diff_two_entities",
    }

def build_scope_switch_chain(subj: str, pid: str, y1: int, v1: str, y2: int, v2: str, truth_type: str) -> list[dict]:
    q1 = explicit_question(pid, subj, y1, truth_type)
    q2 = followup_switch_year(pid, y2)
    return [
        {"question": q1, "answer": v1, "family": "scope_switch"},
        {"question": q2, "answer": v2, "family": "scope_switch"},
    ]

def year_level_paths(snapshot_year: int, truth_type: str) -> dict:
    ydir = out_dir / f"snapshot_year={snapshot_year}" / f"truth_type={truth_type}" / "cross_anchor"
    ydir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": ydir,
        "chains_jsonl": ydir / "chains.jsonl",
        "stats": ydir / "stats.json",
        "done": ydir / "_DONE",
    }
def followup_as_of_anchor(pid: str, anchor_date: str) -> str:
    spec = PID_SPECS[pid]
    prop = spec["surface"]
    qtype = spec["qtype"]

    if qtype == "who":
        return pick([
            "As of {ad}, who was the {prop}?",
            "By {ad}, who held the {prop}?",
        ]).format(ad=anchor_date, prop=prop)

    if qtype == "what":
        if pid == "P31":
            return pick([
                "As of {ad}, what type of thing was it?",
                "By {ad}, what kind of thing was it?",
            ]).format(ad=anchor_date)
        return pick([
            "As of {ad}, what was the {prop}?",
            "By {ad}, what {prop} did it have?",
        ]).format(ad=anchor_date, prop=prop)

    if qtype == "which":
        return pick([
            "As of {ad}, which organization was the {prop}?",
            "By {ad}, which organization was the {prop}?",
        ]).format(ad=anchor_date, prop=prop)

    if qtype == "where":
        return pick([
            "As of {ad}, where was it located?",
            "By {ad}, where was it located?",
        ]).format(ad=anchor_date)

    return f"As of {anchor_date}, what was the {prop}?"

BRIDGE_RELATIONS = {
    "P108": ["P159", "P17"],      # employer → HQ → country
    "P54": ["P131", "P17"],       # team → city → country
    "P127": ["P169", "P17"],      # owned by → CEO; or → country
}

def main():
    import argparse
    import json
    import random
    import time
    import uuid
    from pathlib import Path
    from collections import defaultdict
    from typing import Any, Dict, List, Set, Tuple

    import polars as pl
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--truth_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label_parquet", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--anchor_dates", nargs="*", default=None)
    parser.add_argument("--truth_types", nargs="*", default=None)
    parser.add_argument("--present_day_snapshot_year", type=int, default=2025)
    parser.add_argument("--present_day_anchor_date", type=str, default=None)
    parser.add_argument("--disable_present_day_refs", action="store_true")
    parser.add_argument("--skip_per_anchor", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # caps
    parser.add_argument("--max_rows_temporal", type=int, default=5_000_000)
    parser.add_argument("--max_rows_timeless", type=int, default=1_500_000)

    # flush
    parser.add_argument("--flush_every", type=int, default=50_000)

    # chain controls (within-anchor)
    parser.add_argument("--max_multi_turn_chains", type=int, default=50_000)
    parser.add_argument("--min_chain_len", type=int, default=3)
    parser.add_argument("--max_chain_len", type=int, default=6)

    # multihop/complex caps
    parser.add_argument("--max_multihop2", type=int, default=150_000)
    parser.add_argument("--max_multihop3", type=int, default=50_000)
    parser.add_argument("--max_complex", type=int, default=100_000)

    # backward compat knobs (your older run flags)
    parser.add_argument("--max_complex_count", type=int, default=None)
    parser.add_argument("--max_complex_entity_pair", type=int, default=None)

    # scope switch (within anchor)
    parser.add_argument("--max_scope_switch", type=int, default=50_000)
    parser.add_argument("--max_temporal_narrative", type=int, default=60_000)
    parser.add_argument("--max_change_point", type=int, default=60_000)

    # cross-anchor
    parser.add_argument("--build_cross_anchor", action="store_true")
    parser.add_argument("--no_cross_anchor", action="store_true")
    parser.add_argument("--max_cross_anchor_chains", type=int, default=50_000)
    parser.add_argument("--cross_anchor_per_pair", type=int, default=20_000)

    # debugging
    parser.add_argument("--only_year", type=int, default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    allowed_anchor_dates = set(args.anchor_dates) if args.anchor_dates else None
    allowed_truth_types = set(args.truth_types) if args.truth_types else None

    # Backward compatibility with older flags
    if args.max_complex_count is not None or args.max_complex_entity_pair is not None:
        vals = [v for v in [args.max_complex_count, args.max_complex_entity_pair] if v is not None]
        if vals:
            args.max_complex = min(vals)

    if args.no_cross_anchor:
        args.build_cross_anchor = False

    truth_root = Path(args.truth_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_parquet = Path(args.label_parquet)
    present_day_cache: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}

    def anchor_paths(snapshot_year: int, truth_type: str, anchor_date: str) -> Dict[str, Path]:
        adir = out_dir / f"snapshot_year={snapshot_year}" / f"truth_type={truth_type}" / f"anchor={anchor_date}"
        adir.mkdir(parents=True, exist_ok=True)
        return {
            "dir": adir,
            "qa_jsonl": adir / "data.jsonl",
            "chains_jsonl": adir / "chains.jsonl",
            "stats": adir / "stats.json",
            "done": adir / "_DONE",
        }

    def flush_jsonl(path: Path, buf: List[Dict[str, Any]]):
        if not buf:
            return
        with open(path, "a", encoding="utf-8") as f:
            for x in buf:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        buf.clear()

    def difficulty_single(family: str, truth_type: str) -> Dict[str, Any]:
        analytic = family.startswith("complex_")
        hops = 1
        if family == "multihop_2":
            hops = 2
        elif family == "multihop_3":
            hops = 3
        return {"hops": hops, "temporal": (truth_type == "temporal"), "analytic": analytic}

    def difficulty_chain(nturns: int, family: str, truth_type: str) -> Dict[str, Any]:
        return {"hops": nturns, "temporal": (truth_type == "temporal"), "analytic": False}

    manifest: List[Dict[str, Any]] = []

    # ----------------------------
    # Per-snapshot-year processing
    # ----------------------------
    for snapshot_year in args.years:
        if args.only_year is not None and snapshot_year != args.only_year:
            continue
        print(f"\n[year] {snapshot_year}")

        anchors = list(iter_truth_anchors(truth_root, snapshot_year))
        if allowed_anchor_dates is not None:
            anchors = [x for x in anchors if x[1] in allowed_anchor_dates]
        if allowed_truth_types is not None:
            anchors = [x for x in anchors if x[0] in allowed_truth_types]
        if not anchors:
            raise FileNotFoundError(f"No anchors found for snapshot_year={snapshot_year} under {truth_root / f'snapshot_year={snapshot_year}'}")
        if not args.skip_per_anchor and len(anchors) > 1:
            raise ValueError(
                "Stage 3 now expects per-anchor sharding for within-anchor generation. "
                "Use --anchor_dates to target a single anchor or submit via run_stage3_per_anchor_array.slurm."
            )

        # group anchors by truth_type for cross-anchor later
        anchors_by_type: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
        for ttype, adate, tpath in anchors:
            anchors_by_type[ttype].append((adate, tpath))

        # ----------------------------
        # Per-anchor processing
        # ----------------------------
        if not args.skip_per_anchor:
            for truth_type, anchor_date, truth_path in anchors:
                paths = anchor_paths(snapshot_year, truth_type, anchor_date)

                if paths["done"].exists():
                    print(f"[skip] {snapshot_year} {truth_type} {anchor_date} already done")
                    manifest.append({
                        "year": snapshot_year,
                        "truth_type": truth_type,
                        "anchor_date": anchor_date,
                        "qa_path": str(paths["qa_jsonl"]),
                        "chains_path": str(paths["chains_jsonl"]),
                    })
                    continue

                print(f"[load] year={snapshot_year} type={truth_type} anchor={anchor_date} file={truth_path.name}")
                t0 = time.time()

                df = pl.read_parquet(truth_path)
                df = cap_df(df, truth_type, args.max_rows_temporal, args.max_rows_timeless, args.seed)

            need_labels = ("subject_label" not in df.columns) or ("value_label" not in df.columns)
            if need_labels:
                qids = collect_qids(df)
                label_map = load_en_labels_from_parquet(label_parquet, qids)
                df = attach_labels(df, label_map)
                df = drop_unlabeled(df)

            # hygiene
            if "subject_label" in df.columns:
                df = df.filter(pl.col("subject_label").map_elements(is_clean_label, return_dtype=pl.Boolean))
            if "value_label" in df.columns:
                df = df.filter(pl.col("value_label").map_elements(is_clean_label, return_dtype=pl.Boolean))

            rows = df.to_dicts()
            random.shuffle(rows)

            rich_rows = rows
            rich_truth_path = all_active_truth_path(truth_path)
            if rich_truth_path != truth_path and rich_truth_path.exists():
                try:
                    rich_rows = load_anchor_rows(
                        truth_path=rich_truth_path,
                        snapshot_year=snapshot_year,
                        truth_type=truth_type,
                        label_parquet=label_parquet,
                        max_rows_temporal=args.max_rows_temporal,
                        max_rows_timeless=args.max_rows_timeless,
                        seed=args.seed,
                    )
                except Exception:
                    rich_rows = rows

            # indexes
            by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            by_subject_pid: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
            by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            rich_by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            rich_by_subject_pid: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

            for r in rows:
                pid = r.get("pid")
                if pid not in PID_SPECS:
                    continue

                subj = r.get("subject_label")
                val = r.get("value_label")
                if not is_clean_label(subj) or not is_clean_label(val):
                    continue

                # default partition year
                r["year"] = snapshot_year

                by_subject[subj].append(r)
                by_subject_pid[(subj, pid)].append(r)
                by_pid[pid].append(r)

            for r in rich_rows:
                pid = r.get("pid")
                if pid not in PID_SPECS:
                    continue
                subj = r.get("subject_label")
                val = r.get("value_label")
                if not is_clean_label(subj) or not is_clean_label(val):
                    continue
                r["year"] = snapshot_year
                rich_by_subject[subj].append(r)
                rich_by_subject_pid[(subj, pid)].append(r)

            qa_buf: List[Dict[str, Any]] = []
            chain_buf: List[Dict[str, Any]] = []
            seen_flat: Set[Tuple[int, str, str, str]] = set()
            stats: Dict[str, int] = defaultdict(int)

            if not args.disable_present_day_refs and truth_type not in present_day_cache:
                present_day_cache[truth_type] = build_present_day_lookup(
                    truth_root=truth_root,
                    label_parquet=label_parquet,
                    truth_type=truth_type,
                    present_day_snapshot_year=args.present_day_snapshot_year,
                    present_day_anchor_date=args.present_day_anchor_date,
                    max_rows_temporal=args.max_rows_temporal,
                    max_rows_timeless=args.max_rows_timeless,
                    seed=args.seed,
                )
            present_day_lookup = present_day_cache.get(truth_type, {})

            def annotate_with_present_day(obj: Dict[str, Any]) -> Dict[str, Any]:
                subj = obj.get("subject_label")
                pid = obj.get("pid")
                if subj and pid:
                    pd_row = present_day_lookup.get((subj, pid))
                    if pd_row:
                        obj["present_day_answer"] = pd_row.get("value_label")
                        obj["present_day_value_qid"] = pd_row.get("value_qid")
                        obj["drift_eligible"] = pd_row.get("value_label") != obj.get("answer")
                    else:
                        obj["present_day_answer"] = None
                        obj["present_day_value_qid"] = None
                        obj["drift_eligible"] = False
                return obj

            def emit_flat(qa: Dict[str, Any]) -> bool:
                if not qa_is_high_quality(qa.get("question", ""), qa.get("answer", "")):
                    return False
                qnorm = " ".join(qa["question"].strip().lower().split())
                key = (snapshot_year, anchor_date, truth_type, qnorm)
                if key in seen_flat:
                    return False
                seen_flat.add(key)

                qa["id"] = str(uuid.uuid4())
                qa["snapshot_year"] = snapshot_year
                qa["anchor_date"] = anchor_date
                qa["truth_type"] = truth_type
                qa["difficulty"] = difficulty_single(qa["family"], truth_type)
                annotate_with_present_day(qa)

                qa_buf.append(qa)
                return True

            def emit_chain(turns: List[Dict[str, Any]], family: str):
                if not chain_is_high_quality(turns):
                    return
                obj = {
                    "chain_id": str(uuid.uuid4()),
                    "family": family,
                    "snapshot_year": snapshot_year,
                    "anchor_date": anchor_date,
                    "truth_type": truth_type,
                    "difficulty": difficulty_chain(len(turns), family, truth_type),
                    "turns": [],
                }
                for i, t in enumerate(turns):
                    turn_obj = {
                        "turn_id": str(uuid.uuid4()),
                        "turn_index": i,
                        "question": t["question"],
                        "answer": t["answer"],
                        "year": t.get("year", snapshot_year),
                        "pid": t.get("pid"),
                        "subject_label": t.get("subject_label"),
                        "subject_qid": t.get("subject_qid"),
                        "value_label": t.get("value_label"),
                        "value_qid": t.get("value_qid"),
                    }
                    obj["turns"].append(annotate_with_present_day(turn_obj))
                chain_buf.append(obj)

            # -----------------------
            # A) DIRECT (flat)
            # -----------------------
            for (subj, pid), group in tqdm(list(by_subject_pid.items()), desc=f"direct {snapshot_year} {truth_type} {anchor_date}"):
                r = random.choice(group)
                qa = build_direct(r, truth_type)
                qa["year"] = snapshot_year
                if emit_flat(qa):
                    stats["flat_total"] += 1
                    stats["flat_direct"] += 1
                if args.flush_every > 0 and stats["flat_total"] % args.flush_every == 0:
                    flush_jsonl(paths["qa_jsonl"], qa_buf)
            flush_jsonl(paths["qa_jsonl"], qa_buf)

            # -----------------------
            # B) CARRYOVER (2 turns) + CARRYOVER_THEN (2 turns)
            # -----------------------
            made = 0
            for (subj, pid1), group in tqdm(list(by_subject_pid.items()), desc=f"carryover {snapshot_year} {truth_type} {anchor_date}"):
                pid2 = COMPLEMENTS.get(pid1)
                if not pid2:
                    continue
                if (subj, pid2) not in by_subject_pid:
                    continue

                r1 = random.choice(group)
                r2 = random.choice(by_subject_pid[(subj, pid2)])
                y = snapshot_year

                t1 = turn_from_row(explicit_question(pid1, subj, y, truth_type), r1, year=y)
                t2 = turn_from_row(followup_prop_only(pid2), r2, year=y)

                emit_chain([t1, t2], family="carryover")
                stats["chains_carryover"] += 1

                if truth_type != "timeless":
                    t2b = dict(t2)
                    t2b["question"] = followup_then(pid2)
                    emit_chain([t1, t2b], family="carryover_then")
                    stats["chains_carryover_then"] += 1

                made += 1
                if made % 5000 == 0:
                    flush_jsonl(paths["chains_jsonl"], chain_buf)
            flush_jsonl(paths["chains_jsonl"], chain_buf)

            # -----------------------
            # C) CROSS_ENTITY_THEN (2 turns)
            # -----------------------
            for pid, group in tqdm(list(by_pid.items()), desc=f"cross-entity {snapshot_year} {truth_type} {anchor_date}"):
                if len(group) < 2:
                    continue
                random.shuffle(group)
                cap = min(len(group) - 1, 2000)
                for i in range(0, cap, 2):
                    r1 = group[i]
                    r2 = group[i + 1]
                    if r1["subject_label"] == r2["subject_label"]:
                        continue
                    turns = build_cross_entity_then(r1, r2, truth_type)
                    if not turns:
                        continue
                    emit_chain(turns, family="cross_entity_then")
                    stats["chains_cross_entity_then"] += 1
                flush_jsonl(paths["chains_jsonl"], chain_buf)

            # -----------------------
            # D) SCOPE-SWITCH (2 turns) within anchor (qualifier years)
            # -----------------------
            if truth_type == "temporal":
                made = 0
                by_subj_pid_year = defaultdict(list)

                for r in rows:
                    pid = r.get("pid")
                    if pid not in PID_SPECS:
                        continue
                    subj = r.get("subject_label")
                    val = r.get("value_label")
                    if not subj or not val:
                        continue

                    yq = extract_year_from_qualifiers(r)  # must read pit_date/start_date/end_date
                    if yq is None:
                        continue
                    by_subj_pid_year[(subj, pid, yq)].append(r)

                subjpid_years = defaultdict(set)
                for (subj, pid, yq) in by_subj_pid_year.keys():
                    subjpid_years[(subj, pid)].add(yq)

                candidates = [(subj, pid) for (subj, pid), ys in subjpid_years.items() if len(ys) >= 2]
                random.shuffle(candidates)

                for (subj, pid) in tqdm(candidates, desc=f"scope-switch {snapshot_year} {truth_type} {anchor_date}"):
                    ys = list(subjpid_years[(subj, pid)])
                    if len(ys) < 2:
                        continue

                    y1, y2 = random.sample(ys, 2)
                    r1 = random.choice(by_subj_pid_year[(subj, pid, y1)])
                    r2 = random.choice(by_subj_pid_year[(subj, pid, y2)])

                    t1 = turn_from_row(explicit_question(pid, subj, y1, truth_type), r1, year=y1)
                    t2 = turn_from_row(followup_switch_year(pid, y2), r2, year=y2)

                    emit_chain([t1, t2], family="scope_switch")
                    stats["chains_scope_switch"] += 1

                    made += 1
                    if reached_cap(made, args.max_scope_switch):
                        break
                    if made % 5000 == 0:
                        flush_jsonl(paths["chains_jsonl"], chain_buf)

                flush_jsonl(paths["chains_jsonl"], chain_buf)

            # -----------------------
            # E) MULTI_TURN_CHAIN (4–6 turns)
            # -----------------------
            candidates = []
            for subj, rows in by_subject.items():
                # Count distinct clean PIDs
                clean_pids = set()
                for r in rows:
                    if r.get("pid") in PID_SPECS and is_clean_label(r.get("value_label")):
                        clean_pids.add(r["pid"])
                if len(clean_pids) >= args.min_chain_len:
                    candidates.append(subj)
            
            random.shuffle(candidates)
            # random.shuffle(candidates)

            made_mt = 0
            attempted = 0
            max_attempts = len(candidates) if not cap_enabled(args.max_multi_turn_chains) else min(len(candidates), max(args.max_multi_turn_chains * 50, 100_000))

            pbar = tqdm(total=max_attempts, desc=f"multi-turn {snapshot_year} {truth_type} {anchor_date}")
            for subj in candidates[:max_attempts]:
                attempted += 1
                pbar.update(1)

                turns = build_multi_turn_chain_for_subject(
                    subj=subj,
                    year=snapshot_year,
                    truth_type=truth_type,
                    by_subject_pid=by_subject_pid,
                    min_len=args.min_chain_len,
                    max_len=args.max_chain_len,
                    then_prob=0.35,
                )
                if not turns:
                    continue

                emit_chain(turns, family="multi_turn_chain")
                stats["chains_multi_turn_chain"] += 1
                made_mt += 1

                if made_mt % 5000 == 0:
                    flush_jsonl(paths["chains_jsonl"], chain_buf)

                if reached_cap(made_mt, args.max_multi_turn_chains):
                    break

            pbar.close()
            flush_jsonl(paths["chains_jsonl"], chain_buf)
            print(f"[multi-turn] candidates={len(candidates):,} attempted={attempted:,} made={made_mt:,} success={(made_mt/max(attempted,1))*100:.2f}%")

            # -----------------------
            # E2) TEMPORAL_NARRATIVE / CHANGE_POINT
            # -----------------------
            if truth_type == "temporal":
                narrative_made = 0
                change_point_made = 0
                subj_pid_candidates = list(rich_by_subject_pid.items())
                random.shuffle(subj_pid_candidates)

                for (subj, pid), group in tqdm(subj_pid_candidates, desc=f"temporal-families {snapshot_year} {truth_type} {anchor_date}"):
                    if not reached_cap(narrative_made, args.max_temporal_narrative):
                        chain = build_temporal_narrative_chain(
                            subj=subj,
                            pid=pid,
                            rows=group,
                            min_len=max(3, args.min_chain_len),
                            max_len=max(args.min_chain_len, min(args.max_chain_len + 1, 6)),
                        )
                        if chain:
                            emit_chain(chain, family="temporal_narrative")
                            stats["chains_temporal_narrative"] += 1
                            narrative_made += 1

                    if not reached_cap(change_point_made, args.max_change_point):
                        pid2 = COMPLEMENTS.get(pid)
                        if pid2 and (subj, pid2) in rich_by_subject_pid:
                            cp_chain = build_change_point_chain(
                                subj=subj,
                                pid_main=pid,
                                pid_follow=pid2,
                                rows_main=group,
                                rows_follow=rich_by_subject_pid[(subj, pid2)],
                            )
                            if cp_chain:
                                emit_chain(cp_chain, family="change_point")
                                stats["chains_change_point"] += 1
                                change_point_made += 1

                    if (narrative_made and narrative_made % 5000 == 0) or (change_point_made and change_point_made % 5000 == 0):
                        flush_jsonl(paths["chains_jsonl"], chain_buf)

                    if reached_cap(narrative_made, args.max_temporal_narrative) and reached_cap(change_point_made, args.max_change_point):
                        break

                flush_jsonl(paths["chains_jsonl"], chain_buf)

            # -----------------------
            # F) MULTIHOP_2 / MULTIHOP_3 (flat)
            # -----------------------
            by_entity = by_subject

            multihop2_made = 0
            first_hops = {"P108", "P54", "P131", "P127", "P31"}
            second_hops_by_first = {
                "P108": {"P17", "P127"},
                "P54": {"P17"},
                "P131": {"P17"},
                "P127": {"P169", "P749"},
                "P31": {"P279", "P361"},
            }

            for subj, group in tqdm(list(by_subject.items()), desc=f"multihop-2 {snapshot_year} {truth_type} {anchor_date}"):
                for r1 in group:
                    pid1 = r1.get("pid")
                    if pid1 not in first_hops:
                        continue
                    mid = r1.get("value_label")
                    if not is_clean_label(mid):
                        continue
                    want = second_hops_by_first.get(pid1, set())
                    for r2 in by_entity.get(mid, []):
                        if r2.get("pid") in want and is_clean_label(r2.get("value_label")):
                            qa = build_multihop_2(r1, r2, truth_type)
                            qa["year"] = snapshot_year
                            if emit_flat(qa):
                                stats["flat_total"] += 1
                                stats["flat_multihop_2"] += 1
                                multihop2_made += 1
                            break
                    if reached_cap(multihop2_made, args.max_multihop2):
                        break
                if reached_cap(multihop2_made, args.max_multihop2):
                    break

            flush_jsonl(paths["qa_jsonl"], qa_buf)

            multihop3_made = 0
            for subj, group in tqdm(list(by_subject.items()), desc=f"multihop-3 {snapshot_year} {truth_type} {anchor_date}"):
                for r1 in group:
                    if r1.get("pid") != "P108":
                        continue
                    org = r1.get("value_label")
                    if not is_clean_label(org):
                        continue

                    parent_row = None
                    for r2 in by_entity.get(org, []):
                        if r2.get("pid") == "P127" and is_clean_label(r2.get("value_label")):
                            parent_row = r2
                            break
                    if not parent_row:
                        continue

                    parent = parent_row["value_label"]
                    ceo_row = None
                    for r3 in by_entity.get(parent, []):
                        if r3.get("pid") == "P169" and is_clean_label(r3.get("value_label")):
                            ceo_row = r3
                            break
                    if not ceo_row:
                        continue

                    qa = build_multihop_3(r1, parent_row, ceo_row, truth_type)
                    qa["year"] = snapshot_year
                    if emit_flat(qa):
                        stats["flat_total"] += 1
                        stats["flat_multihop_3"] += 1
                        multihop3_made += 1

                    if reached_cap(multihop3_made, args.max_multihop3):
                        break
                if reached_cap(multihop3_made, args.max_multihop3):
                    break

            flush_jsonl(paths["qa_jsonl"], qa_buf)

            # -----------------------
            # G) COMPLEX arithmetic (flat)
            # -----------------------
            complex_made = 0
            arith_pids = ["P108", "P54", "P39", "P463", "P69"]

            for subj, srows in tqdm(list(rich_by_subject.items()), desc=f"complex {snapshot_year} {truth_type} {anchor_date}"):
                if reached_cap(complex_made, args.max_complex):
                    break
                for pid in arith_pids:
                    qa = build_count_distinct_in_year(srows, pid, snapshot_year, truth_type)
                    if qa:
                        qa["year"] = snapshot_year
                        if emit_flat(qa):
                            stats["flat_total"] += 1
                            stats[qa["family"]] += 1
                            complex_made += 1
                            if reached_cap(complex_made, args.max_complex):
                                break

            flush_jsonl(paths["qa_jsonl"], qa_buf)

            if truth_type != "timeless":
                cands_by_pid: Dict[str, List[str]] = defaultdict(list)
                for (subj, pid), group in rich_by_subject_pid.items():
                    if pid in arith_pids:
                        vals = _distinct_values_in_year(group, pid, snapshot_year)
                        if len(vals) >= 1:
                            cands_by_pid[pid].append(subj)

                for pid in arith_pids:
                    if reached_cap(complex_made, args.max_complex):
                        break
                    cands = list(set(cands_by_pid.get(pid, [])))
                    if len(cands) < 2:
                        continue
                    random.shuffle(cands)

                    for i in range(0, min(len(cands) - 1, 20000), 2):
                        e1 = cands[i]
                        e2 = cands[i + 1]

                        qa1 = build_sum_two_entities(rich_by_subject[e1], rich_by_subject[e2], pid, snapshot_year, truth_type)
                        if qa1:
                            qa1["year"] = snapshot_year
                            if emit_flat(qa1):
                                stats["flat_total"] += 1
                                stats[qa1["family"]] += 1
                                complex_made += 1

                        qa2 = build_diff_two_entities(rich_by_subject[e1], rich_by_subject[e2], pid, snapshot_year, truth_type)
                        if qa2:
                            qa2["year"] = snapshot_year
                            if emit_flat(qa2):
                                stats["flat_total"] += 1
                                stats[qa2["family"]] += 1
                                complex_made += 1

                        if reached_cap(complex_made, args.max_complex):
                            break

                flush_jsonl(paths["qa_jsonl"], qa_buf)

            # -------- write stats + done
            flush_jsonl(paths["qa_jsonl"], qa_buf)
            flush_jsonl(paths["chains_jsonl"], chain_buf)

            stats_out = dict(stats)
            stats_out["elapsed_sec"] = round(time.time() - t0, 3)
            with open(paths["stats"], "w", encoding="utf-8") as f:
                json.dump(stats_out, f, indent=2, ensure_ascii=False)

            paths["done"].write_text("ok\n", encoding="utf-8")

            manifest.append({
                "year": snapshot_year,
                "truth_type": truth_type,
                "anchor_date": anchor_date,
                "qa_path": str(paths["qa_jsonl"]),
                "chains_path": str(paths["chains_jsonl"]),
            })

            print(f"[done] {snapshot_year} {truth_type} {anchor_date} flat={stats.get('flat_total', 0):,}")

        # ----------------------------
        # Cross-anchor pass (temporal only)
        # ----------------------------
        if args.build_cross_anchor:
            truth_type = "temporal"
            anchors_tt = sorted(anchors_by_type.get(truth_type, []), key=lambda x: x[0])
            if len(anchors_tt) >= 2:
                anchor_indexes: List[Tuple[str, Dict[Tuple[str, str], List[Dict[str, Any]]]]] = []
                for adate, tpath in anchors_tt:
                    rows2 = load_anchor_rows(
                        truth_path=tpath,
                        snapshot_year=snapshot_year,
                        truth_type=truth_type,
                        label_parquet=label_parquet,
                        max_rows_temporal=args.max_rows_temporal,
                        max_rows_timeless=args.max_rows_timeless,
                        seed=args.seed,
                    )
                    idx = defaultdict(list)
                    for r in rows2:
                        pid = r.get("pid")
                        subj = r.get("subject_label")
                        val = r.get("value_label")
                        if pid in PID_SPECS and is_clean_label(subj) and is_clean_label(val):
                            idx[(subj, pid)].append(r)
                    anchor_indexes.append((adate, idx))

                cross_dir = out_dir / f"snapshot_year={snapshot_year}" / f"truth_type={truth_type}" / "cross_anchor"
                cross_dir.mkdir(parents=True, exist_ok=True)
                cross_path = cross_dir / "chains.jsonl"

                made = 0
                with open(cross_path, "w", encoding="utf-8") as f:
                    for i in range(len(anchor_indexes) - 1):
                        adate_a, idx_a = anchor_indexes[i]
                        adate_b, idx_b = anchor_indexes[i + 1]

                        subjects_a = set(s for (s, _) in idx_a.keys())
                        subjects_b = set(s for (s, _) in idx_b.keys())
                        common_subjects = list(subjects_a.intersection(subjects_b))
                        random.shuffle(common_subjects)

                        made_pair = 0
                        for subj in common_subjects:
                            # Dynamically build a plan: any PID available in BOTH anchors
                            pids_a = {pid for (s, pid) in idx_a.keys() if s == subj and pid in PID_SPECS}
                            pids_b = {pid for (s, pid) in idx_b.keys() if s == subj and pid in PID_SPECS}
                            common_pids = list(pids_a & pids_b)
                            
                            if len(common_pids) < 2:  # need at least 2 for a chain
                                continue
                                
                            # Shuffle and take 2–4 PIDs
                            random.shuffle(common_pids)
                            plan = common_pids[:random.randint(2, min(4, len(common_pids)))]
                            
                            chain = build_cross_anchor_chain(
                                subj=subj,
                                plan=plan,
                                year_a=snapshot_year,
                                year_b=snapshot_year,
                                truth_type=truth_type,
                                by_subject_pid_a=idx_a,
                                by_subject_pid_b=idx_b,
                                then_prob=0.35,
                            )
                            if not chain:
                                continue

                            obj = {
                                "chain_id": str(uuid.uuid4()),
                                "family": "cross_anchor_switch",
                                "snapshot_year": snapshot_year,
                                "truth_type": truth_type,
                                "anchor_from": adate_a,
                                "anchor_to": adate_b,
                                "difficulty": {"hops": len(chain), "temporal": True, "analytic": False},
                                "turns": [
                                    {"turn_id": str(uuid.uuid4()), "turn_index": j, **t}
                                    for j, t in enumerate(chain)
                                ],
                            }
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            made += 1
                            made_pair += 1

                            if reached_cap(made_pair, args.cross_anchor_per_pair):
                                break
                            if reached_cap(made, args.max_cross_anchor_chains):
                                break

                        if reached_cap(made, args.max_cross_anchor_chains):
                            break

                print(f"[cross-anchor] year={snapshot_year} wrote {made:,} chains -> {cross_path}")

    manifest_path = out_dir / "stage3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[manifest] wrote {len(manifest)} anchor entries -> {manifest_path}")


if __name__ == "__main__":
    main()
