from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from crysreas import Config
from crysreas.data.prompt_generator import get_info
from pathlib import Path
import json
import shelve
from typing import List
from crysreas.utils.crystal import SimpleCrystal
import re, csv
from tqdm import tqdm
import tyro
from dataclasses import dataclass

def main(
    split: List[str] = None,
    ckpt_path: Path = Config.PROJECT_ROOT / "checkpoints_merged" / "rl_thinking_mix/",
    prompt_type: str = "conditional+thinking",
    db_path: Path = Config.DATA_PATH / "MP_shelve",
    out_path: Path = Path(Config.PROJECT_ROOT / "generation_results.csv"),
    batch_size = 4,
    input_max_length = 2048,
    output_max_length = 2048
):
    if(split == None):
        split = Path(Config.DATA_PATH / "split.json")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if(isinstance(split, Path)):
        with open(split, "r") as f:
            split = json.load(f)["test"]
            
    db = shelve.open(str(db_path))

    print("split", split)
    print(SimpleCrystal(db[split[0]]["structure"]).to_simple_no_sym())
    
    results_to_csv = []
    for i in tqdm(range(0, len(split), batch_size)):
        batch_prompts = []
        batch_keys = []
        for j in range(i, min(len(split), i + batch_size)):
            info = get_info(db[split[j]], prompt_type)
            prompt = info["question"]
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(text)
            batch_keys.append(split[j])

        model_inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left",
            truncation=True, 
            max_length=input_max_length
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=output_max_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        batch_gen_strs = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        for gen_str, key in zip(batch_gen_strs, batch_keys):
            print("generated", gen_str, "ground_truth", SimpleCrystal(db[key]["structure"]).to_simple_no_sym())
            match = re.search(r'<CIF>(.*?)</CIF>', gen_str, re.DOTALL)
            cif_simple = None
            if match:
                cif_simple = match.group(1).strip()
            #     try:
            #         cif = SimpleCrystal.from_simple(cif_simple).to_cif()
            #     except:
            #         cif = ""
            # else:
            #     cif = ""
            
            results_to_csv.append({
                'gen_str': gen_str,
                "cif_simple": cif_simple
            })

    print(f"\nWriting results to CSV file: {out_path}")
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['gen_str', 'cif_simple']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results_to_csv)

if __name__ == "__main__":
    tyro.cli(main)