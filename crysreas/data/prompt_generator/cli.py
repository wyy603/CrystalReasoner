import shelve
from dataclasses import dataclass
import pathlib

import torch
import tyro

from verl.utils import hf_tokenizer

from crysreas import Config

from .get_info import get_info


@dataclass
class Args:
    db_path: pathlib.Path = Config.DATA_PATH / "MP_shelve"
    key: str = "mp-1"
    debug: bool = False
    prompt_type: str = "conditional+thinking"
    seed: int = 42


def main(args: Args):
    db_path = args.db_path
    key = args.key
    debug = args.debug
    prompt_type = args.prompt_type
    seed = args.seed
    
    db = shelve.open(str(db_path), "r")
    elem = db[key]
    info = get_info(elem, prompt_type, debug, seed)

    tokenizer = hf_tokenizer("MegaScience/Qwen2.5-3B-MegaScience", trust_remote_code=True)
    prompt = info["question"]
    response = info["answer"]

    prompt_chat = [{"role": "user", "content": prompt}]
    prompt_chat_str = tokenizer.apply_chat_template(
        prompt_chat, add_generation_prompt=True, tokenize=False
    )
    response_chat_str = response + tokenizer.eos_token
    print(prompt_chat_str + response_chat_str)

    prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_ids_output["input_ids"][0]
    response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
    response_ids = response_ids_output["input_ids"][0]
    input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    print("Token Count:", input_ids.shape[0])


# now mp-1221227
# mp-555507
# mp-1
# mp-1181546
# mp-1181657


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    main(cli_args)
