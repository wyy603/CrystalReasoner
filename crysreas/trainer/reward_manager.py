import hydra
import ray
import torch
import os
import psutil
from omegaconf import OmegaConf
import numpy as np
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import gsm8k, math_reward
import re
from crysreas.trainer.crystal_dataset_rl import CrystalDatasetRL
from crysreas.utils.crystal import SimpleCrystal
from pymatgen.core.structure import Structure
from crysreas.metric_process import MetricProcess
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=Warning, module=r"mattergen(\.|$)")
warnings.filterwarnings("ignore", category=Warning, module=r"uncertainties(\.|$)")
warnings.filterwarnings("ignore", category=Warning, module=r"pymatgen(\.|$)")

def to_number(arr):
    arr = np.asanyarray(arr)
    if(arr.dtype == np.object_):
        is_invalid = pd.isna(arr)

        mask_not_zero = np.ones(arr.shape, dtype=bool)
        valid_indices = ~is_invalid
        if valid_indices.any():
            try:
                vals = arr[valid_indices]
                mask_not_zero[valid_indices] = (vals != 0)
            except:
                pass
        
        final_mask = (~is_invalid) & mask_not_zero
        return final_mask.astype(np.float32)
    else:
        arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    return arr

class RewardManager:
    def __init__(self, tokenizer, num_examine, custom_reward_function, custom_data) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.necessary = custom_reward_function["necessary"]
        self.weights = custom_reward_function["weights"]
        self.prompt_type = custom_data["prompt_type"]
        self._logged_prompt_type = False
        _pt = list(self.prompt_type.split("+"))
        # Ray chunk-parallel for non-heavy metrics; heavy (e.g. MLIP) runs in-process (see MetricProcess).
        self._metric_proc = MetricProcess({"prompt_type": _pt})
    
    def __call__(self, data: DataProto, return_dict: bool = False, log = False):
        """We will expand this function gradually based on the available datasets"""
        if log and not self._logged_prompt_type:
            print("rm.prompt_type", self.prompt_type)
            self._logged_prompt_type = True

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        valid_response_lengths = []

        process_data = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            prompt_str = self.tokenizer.decode(prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            if log:
                print(f"prompt_str = {prompt_str}\nresponse_str = {response_str}")
            encoding = self.tokenizer(
                response_str, 
                return_offsets_mapping=True, 
                add_special_tokens=False
            )
            offset = np.array(encoding['offset_mapping'])
            gt = data_item.non_tensor_batch["gt"]

            process_data.append({
                "responses": response_str,
                "offset": offset,
                "gt": gt
            })

        # check for necessary (metric_process: DataFrame pipeline + optional Ray workers)
        process_data = pd.DataFrame(process_data)
        valid = np.ones((len(data),), dtype=np.bool_)
        process = psutil.Process(os.getpid())
        necessary_names = list(self.necessary.keys())
        if log:
            print("necessary_names: ", necessary_names)
        if necessary_names:
            process_data = self._metric_proc.process(
                process_data, necessary_names, forced=False
            )
            for dep_name in necessary_names:
                metric_output = process_data[dep_name].to_numpy()
                valid &= metric_output != 0
                mem_usage = process.memory_info().rss / 1024**3
                if log:
                    print(f"Metric {dep_name} finished. CPU Memory: {mem_usage:.2f} GB", flush=True)
                process_data[f"mem/{dep_name}"] = mem_usage

        rewards = np.zeros((len(data),), dtype=np.float32)
        indices = valid.nonzero()[0]
        subset = process_data.iloc[indices].copy()
        weight_names = list(self.weights.keys())
        if log:
            print("Number of valid structures: ", len(indices))
        if len(indices) > 0 and weight_names:
            subset = self._metric_proc.process(subset, weight_names, forced=False)
            for dep_name in weight_names:
                if log:
                    print("dep_name", dep_name)
                metric_output = subset[dep_name].to_numpy()
                metric_output = to_number(metric_output)
                if log:
                    print(dep_name, metric_output)
                rewards[indices] += float(self.weights[dep_name]) * metric_output
                mem_usage = process.memory_info().rss / 1024**3
                if log:
                    print(f"Metric {dep_name} finished. CPU Memory: {mem_usage:.2f} GB", flush=True)
                process_data.loc[subset.index, f"mem/{dep_name}"] = mem_usage
        new_cols = subset.columns.difference(process_data.columns)
        if len(new_cols) > 0:
            process_data.loc[subset.index, new_cols] = subset[new_cols]

        # assign reward
        for i in range(len(data)):
            if(valid_response_lengths[i] > 0):
                reward_tensor[i, valid_response_lengths[i] - 1] = rewards[i].item()

        # return curves
        data_curves = {f"rewards/{col}": to_number(process_data[col].to_numpy().copy()) for col in process_data.columns}

        del process_data

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": data_curves
            }
        else:
            return reward_tensor
        
from transformers import AutoTokenizer
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
import torch.nn.functional as F
import os
from tqdm import tqdm
import time

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    custom_reward_fn = {
        "mode": "default",
        "necessary": {
            "fit_format": 1,
        },
        "weights": {
            #"simple_structure": 0.1,
            #"structure_validity": 0.3,
            #"smact_validity": 0.3,
            #"composition_consistency": 0.3,
            "cte_reward": 1.0,
            #"elastic_reward_all": 1.0,
        },
    }
    custom_data = {
        "split_path": "assets/MP/split_cte.json",
        "db_path": "assets/MP/MP_shelve",
        "prompt_type": "cte+thinking"
    }
    num_samples = 1

    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints_merged/20260202/global_step_800", 
        trust_remote_code=True,
    )
    print(config.data.custom_data)
    config.data.shuffle = False
    config.data.custom_data.split_path = custom_data["split_path"]
    config.data.custom_data.db_path = custom_data["db_path"]
    config.data.custom_data.prompt_type = custom_data["prompt_type"]
    dataset = CrystalDatasetRL(config.data.train_files, tokenizer, config.data, None)
    rm = RewardManager(
        tokenizer,
        0,
        custom_reward_function=custom_reward_fn,
        custom_data=custom_data
    )

    # prepare dataset and response strings.
    raw_dataset = []
    for i in range(num_samples):
        raw_dataset.append(dataset[0])
    item = dataset.db[dataset.split[dataset.split_name][0]]
    response_strs = [f"""## CIF File
<CIF>{SimpleCrystal.from_sym_structure(item["structure"]).to_simple_no_sym()}</CIF>"""] * num_samples #mp-1188312
    
    print("Total number of items: ", len(raw_dataset))
    print("Testing items: ", raw_dataset[0]["gt"]["mp_id"])
    for i, item in enumerate(raw_dataset):
        item["prompts"] = item["input_ids"]

        response_str = response_strs[i]

        response = tokenizer(
            response_str + "<|im_end|><|endoftext|>", 
            return_tensors="pt", 
            add_special_tokens=False, 
            padding="max_length", 
            truncation=True,
            max_length=1024
        )

        attention_mask = torch.cat((item["attention_mask"], response["attention_mask"][0]))
        item["attention_mask"] = attention_mask
        item["responses"] = response["input_ids"][0]
    
    train_sampler = create_rl_sampler(config.data, raw_dataset)
    dataloader = StatefulDataLoader(
        dataset=raw_dataset,
        batch_size=num_samples,
        num_workers=1,
        shuffle=False,
        drop_last=False,
        collate_fn=default_collate_fn,
        sampler=train_sampler
    )
    a = time.time()
    for test_data in dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        tensor = rm(test_batch, log = False)
        print("Reward tensor: ", tensor, "Sum of reward tensor: ", tensor.sum(), "Averge Reward: ", tensor.sum() / num_samples)
    b = time.time()
    print("Time taken (s): ", b - a)

if __name__ == "__main__":
    main()
