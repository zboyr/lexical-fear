"""Download candidate datasets and extract a unified prompt pool.

Goal: find datasets (or combinations) where harmful vs benign prompts are
lexically confusable, i.e. a TF-IDF + LogisticRegression source classifier
gets AUC close to 0.5. Target pool size ~5000 harmful + 5000 benign.

Output: data/pool.parquet with columns
  text    - prompt text
  label   - 1 harmful / 0 benign
  source  - dataset name
  subtype - within-dataset subset (e.g. wjb adversarial_harmful)

All datasets here are ungated on HF for this account. allenai/wildjailbreak
is gated (auto-approval); we use the ungated mirror heegyu/wildjailbreak-train,
which preserves the official data_type column.
"""
import os
import re

import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
OUT = os.path.join(POOLS, "pool.parquet")

rows = []


def add(text, label, source, subtype):
    if text is None:
        return
    text = str(text).strip()
    if len(text) < 5:
        return
    rows.append({"text": text, "label": int(label), "source": source, "subtype": subtype})


# --- WildJailbreak (mirror of allenai/wildjailbreak train) -------------------
print("wildjailbreak ...")
ds = load_dataset("heegyu/wildjailbreak-train", split="train")
for ex in ds:
    dt = ex["data_type"]
    label = 1 if dt.endswith("harmful") else 0
    text = ex["adversarial"] if dt.startswith("adversarial") else ex["vanilla"]
    add(text, label, "wjb", dt)

# --- OR-Bench ----------------------------------------------------------------
print("or-bench ...")
for cfg, label, sub in [("or-bench-toxic", 1, "toxic"),
                        ("or-bench-hard-1k", 0, "hard"),
                        ("or-bench-80k", 0, "80k")]:
    for ex in load_dataset("bench-llm/or-bench", cfg, split="train"):
        add(ex["prompt"], label, "orbench", sub)

# --- FalseReject (benign seemingly-toxic; train + test) ----------------------
print("falsereject ...")
for split in ["train", "test"]:
    try:
        for ex in load_dataset("AmazonScience/FalseReject", split=split):
            add(ex["prompt"], 0, "falsereject", split)
    except ValueError:
        pass

# --- PHTest (pseudo-harmful benign; keep 'harmless' only, drop controversial)
print("phtest ...")
for ex in load_dataset("furonghuang-lab/PHTest", split="train"):
    if ex["Harmfulness"] == "harmless":
        add(ex["Request"], 0, "phtest", "harmless")

# --- CoCoNot (safety category: original = should-refuse, contrast = look-alike benign)
print("coconot ...")
SAFETY = "Requests with safety concerns"
for split in ["train", "test"]:
    try:
        for ex in load_dataset("allenai/coconot", "original", split=split):
            if ex["category"] == SAFETY:
                add(ex["prompt"], 1, "coconot", "original_safety")
    except ValueError:
        pass
for ex in load_dataset("allenai/coconot", "contrast", split="test"):
    if ex["category"] == SAFETY:
        add(ex["prompt"], 0, "coconot", "contrast_safety")

# --- XSTest (contrast_* types are unsafe) -------------------------------------
print("xstest ...")
seen = set()
for ex in load_dataset("natolambert/xstest-v2-copy", split="gpt4"):
    if ex["prompt"] in seen:
        continue
    seen.add(ex["prompt"])
    add(ex["prompt"], 1 if ex["type"].startswith("contrast_") else 0, "xstest", ex["type"])

# --- JBB-Behaviors (100 harmful + 100 benign paired) --------------------------
print("jbb ...")
for split, label in [("harmful", 1), ("benign", 0)]:
    for ex in load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=split):
        add(ex["Goal"], label, "jbb", split)

# --- ToxicChat (real traffic; human-annotated) ---------------------------------
print("toxic-chat ...")
for ex in load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train"):
    sub = "jailbreaking" if ex["jailbreaking"] == 1 else ("toxic" if ex["toxicity"] == 1 else "clean")
    add(ex["user_input"], 1 if ex["toxicity"] == 1 or ex["jailbreaking"] == 1 else 0,
        "toxicchat", sub)

# --- Salad-Data (harmful only; base questions + attack-enhanced) ---------------
print("salad ...")
for ex in load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train"):
    add(ex["question"], 1, "salad", "base")
for ex in load_dataset("OpenSafetyLab/Salad-Data", "attack_enhanced_set", split="train"):
    add(ex["augq"], 1, "salad", "attack_enhanced")

# --- ALERT (harmful only; strip the instruction template) ----------------------
print("alert ...")
pat = re.compile(r"^### Instruction:\s*", flags=re.M)
for cfg, sub in [("alert", "plain"), ("alert_adversarial", "adversarial")]:
    for ex in load_dataset("Babelscape/ALERT", cfg, split="test"):
        text = pat.sub("", ex["prompt"]).replace("### Response:", "").strip()
        add(text, 1, "alert", sub)

df = pd.DataFrame(rows)
# global dedup within (source, label) to avoid CV leakage from repeated prompts
df = df.drop_duplicates(subset=["source", "label", "text"]).reset_index(drop=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_parquet(OUT)
print(f"\nsaved {len(df)} rows -> {OUT}")
print(df.groupby(["source", "label", "subtype"]).size())
