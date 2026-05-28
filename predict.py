#!/usr/bin/env python
"""
SynFit-Predict: score protein variants with a trained SynFit 2-head joint model.

Imports the model class from SynFit/train_joint_shared_module.py (mounted at runtime).
Replicates the scoring scheme used during SynFit training:
  - mask the mutated positions in the variant sequence
  - forward through branch A then branch B
  - score = sum over mutated positions of [log p(mutant_aa) - log p(wildtype_aa)]
Writes out/predictions.csv with one column per objective and (optionally) a Pareto rank.
"""

import os
import re
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmConfig

warnings.filterwarnings("ignore")

# SynFit/ is on PYTHONPATH via run.sh, so we can import directly.
from train_joint_shared_module import EsmForMaskedLM_2Head  # noqa: E402

# ---------- Inputs ----------
WT_SEQ        = os.environ.get("wt_sequence", "").strip()
METRIC_NAMES  = [s.strip() for s in os.environ.get("metric_names", "objective_A,objective_B").split(",")]
BATCH_SIZE    = int(os.environ.get("batch_size", "8"))
COMPUTE_PARETO = os.environ.get("compute_pareto", "true").lower() in ("1", "true", "yes")

if not WT_SEQ:
    sys.exit("ERROR: wt_sequence env var is empty")
if len(METRIC_NAMES) != 2:
    print(f"WARNING: expected 2 metric_names, got {METRIC_NAMES}. Padding/truncating.")
    METRIC_NAMES = (METRIC_NAMES + ["objective_A", "objective_B"])[:2]

os.makedirs("out", exist_ok=True)

# ---------- Locate input files ----------
ckpt_candidates = sorted(
    [p for p in glob.glob("inputs/*") if p.lower().endswith((".pth", ".pt", ".bin"))]
)
if not ckpt_candidates:
    sys.exit("ERROR: no checkpoint (.pth/.pt/.bin) found in inputs/")
ckpt_path = ckpt_candidates[0]
print(f"Checkpoint: {ckpt_path}")

csv_candidates = sorted(glob.glob("inputs/*.csv"))
if not csv_candidates:
    sys.exit("ERROR: no variants CSV found in inputs/")
variants_csv = csv_candidates[0]
print(f"Variants CSV: {variants_csv}")

df = pd.read_csv(variants_csv)
if "mutant" not in df.columns:
    sys.exit("ERROR: variants CSV must have a 'mutant' column")
print(f"Loaded {len(df)} variants")

# ---------- Build mutated_sequence if not provided ----------
def apply_mutations(wt: str, mutant_str: str) -> str:
    seq = list(wt)
    for mut in re.split(r"[,:;]", mutant_str.strip()):
        m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", mut.strip())
        if not m:
            raise ValueError(f"Bad mutation token: {mut!r}")
        orig, pos1, new = m.group(1), int(m.group(2)), m.group(3)
        pos0 = pos1 - 1
        if pos0 < 0 or pos0 >= len(seq):
            raise ValueError(f"Position {pos1} out of range for WT length {len(wt)}")
        if seq[pos0] != orig:
            raise ValueError(
                f"Mutation {mut}: expected {orig} at position {pos1}, found {seq[pos0]}"
            )
        seq[pos0] = new
    return "".join(seq)

if "mutated_sequence" not in df.columns:
    df["mutated_sequence"] = df["mutant"].apply(lambda m: apply_mutations(WT_SEQ, m))

def extract_positions_0idx(mutant_str: str) -> list:
    return [int(n) - 1 for n in re.findall(r"\d+", mutant_str)]

df["positions_0idx"] = df["mutant"].apply(extract_positions_0idx)

# ---------- GPU check ----------
if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA GPU available")
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------- Load tokenizer + base config for ESM2-650M ----------
BACKBONE = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(BACKBONE)
config = EsmConfig.from_pretrained(BACKBONE)

# ---------- Build the 2-head model and load the joint state dict ----------
print("Instantiating EsmForMaskedLM_2Head...")
model = EsmForMaskedLM_2Head(config)
# Drop unused submodules the joint training script also deletes - keeps keys aligned.
if hasattr(model.esm, "pooler") and model.esm.pooler is not None:
    del model.esm.pooler
if hasattr(model, "lm_head"):
    del model.lm_head
if hasattr(model.esm, "contact_head"):
    del model.esm.contact_head

print(f"Loading state dict from {ckpt_path}...")
state = torch.load(ckpt_path, map_location="cpu")
# DataParallel saves may have a 'module.' prefix on every key.
if any(k.startswith("module.") for k in state):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
if missing[:5]:
    print("  Example missing:", missing[:5])
if unexpected[:5]:
    print("  Example unexpected:", unexpected[:5])

model.to(device).eval()

# ---------- Tokenize WT ----------
seq_len = len(WT_SEQ)
pad_len = seq_len + 2  # ESM adds <cls>/<eos>

wt_enc = tokenizer([WT_SEQ], padding="max_length", max_length=pad_len, return_tensors="pt")
wt_ids = wt_enc["input_ids"].to(device)  # [1, L+2]

# ---------- Scoring ----------
@torch.no_grad()
def score_batch(seqs, positions_list, branch):
    enc = tokenizer(seqs, padding="max_length", max_length=pad_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    masked = ids.clone()
    mask_id = tokenizer.mask_token_id
    for i, pos in enumerate(positions_list):
        valid = [p for p in pos if 0 <= p < ids.size(1) - 2]
        for p in valid:
            masked[i, 1 + p] = mask_id

    logits = model(input_ids=masked, attention_mask=mask, branch=branch)
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = torch.zeros(ids.size(0), device=device)
    for i, pos in enumerate(positions_list):
        valid = [p for p in pos if 0 <= p < ids.size(1) - 2]
        if not valid:
            continue
        idx = torch.tensor(valid, device=device)
        mt_ids = ids[i, 1 + idx]
        wt_ids_i = wt_ids[0, 1 + idx]
        lp = log_probs[i]
        scores[i] = torch.sum(lp[1 + idx, mt_ids] - lp[1 + idx, wt_ids_i])
    return scores.cpu().numpy()

scores_A, scores_B = [], []
n = len(df)
print(f"Scoring {n} variants in batches of {BATCH_SIZE}...")
for start in tqdm(range(0, n, BATCH_SIZE)):
    end = min(start + BATCH_SIZE, n)
    seqs = df["mutated_sequence"].iloc[start:end].tolist()
    pos  = df["positions_0idx"].iloc[start:end].tolist()
    scores_A.extend(score_batch(seqs, pos, branch="A"))
    scores_B.extend(score_batch(seqs, pos, branch="B"))

df[METRIC_NAMES[0]] = scores_A
df[METRIC_NAMES[1]] = scores_B

# ---------- Optional Pareto ranking (higher is better for both metrics) ----------
if COMPUTE_PARETO:
    print("Computing non-dominated Pareto ranks...")
    a = df[METRIC_NAMES[0]].to_numpy()
    b = df[METRIC_NAMES[1]].to_numpy()
    n = len(df)
    remaining = np.ones(n, dtype=bool)
    rank = np.zeros(n, dtype=int)
    current = 1
    while remaining.any():
        idxs = np.where(remaining)[0]
        on_front = []
        for i in idxs:
            dominated = False
            for j in idxs:
                if j == i:
                    continue
                if (a[j] >= a[i] and b[j] >= b[i]) and (a[j] > a[i] or b[j] > b[i]):
                    dominated = True
                    break
            if not dominated:
                on_front.append(i)
        for i in on_front:
            rank[i] = current
            remaining[i] = False
        current += 1
    df["pareto_rank"] = rank

# ---------- Save ----------
out_cols = ["mutant", "mutated_sequence", METRIC_NAMES[0], METRIC_NAMES[1]]
if COMPUTE_PARETO:
    out_cols.append("pareto_rank")
df[out_cols].to_csv("out/predictions.csv", index=False)
print(f"Wrote out/predictions.csv ({len(df)} rows)")
print("Done.")
