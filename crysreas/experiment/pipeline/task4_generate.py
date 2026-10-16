"""Sample responses for a single MP prompt and dump them in main_generation parquet format."""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
from crysreas.data.prompt_generator import get_info

p = argparse.ArgumentParser()
p.add_argument("--mp-ids", nargs="+", required=True)
p.add_argument("--model", default="checkpoints_merged/rl_thinking_mix")
p.add_argument("--prompt-type", default="conditional+thinking")
p.add_argument("--n", type=int, default=16)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--top-p", type=float, default=0.7)
p.add_argument("--max-new-tokens", type=int, default=3584)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
args = p.parse_args()
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

recs = pickle.load(open(str(DATA_DIR / "test_records.pkl"), "rb"))
prompts = {k: get_info(recs[k], args.prompt_type, seed=k[3:])["question"] for k in args.mp_ids}

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()

torch.manual_seed(args.seed)
rows = []
for mp_id in args.mp_ids:
    chat = [{"role": "user", "content": prompts[mp_id]}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    enc = tok([text] * args.n, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=50,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    responses = [
        tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        for i in range(args.n)
    ]
    rows.append({"mp_id": mp_id, "prompt": chat, "responses": responses,
                 "extra_args": [{"sum_of_information": None, "average_of_information": None,
                                 "token_num": None}] * args.n})
    print(f"[{mp_id}] generated {args.n} samples", flush=True)

pd.DataFrame(rows).to_parquet(args.out)
json.dump(prompts, open(args.out + ".prompts.json", "w"), indent=2)
print("wrote", args.out)
