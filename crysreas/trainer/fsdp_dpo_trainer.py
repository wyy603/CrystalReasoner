import time

import hydra
import torch
from tensordict import TensorDict
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.torch_dtypes import PrecisionType

from crysreas.trainer.dpo_dataset import OfflineDPODataset
from crysreas.trainer.dpo_loss import compute_dpo_loss, get_batch_logps
from crysreas.trainer.fsdp_sft_trainer import FSDPSFTTrainer, logger


class FSDPDPOTrainer(FSDPSFTTrainer):
    """Offline DPO trainer using the existing SFT FSDP/checkpoint stack."""

    def _build_model_optimizer(self):
        super()._build_model_optimizer()
        self._build_reference_model()

    def _build_reference_model(self):
        if self.lora:
            raise NotImplementedError("Offline DPO trainer does not support LoRA adapters yet.")

        reference_path = self.config.model.get("reference_pretrain", self.config.model.partial_pretrain)
        local_reference_path = copy_to_local(src=reference_path, verbose=True)
        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "fp32"))

        ref_config = AutoConfig.from_pretrained(local_reference_path, trust_remote_code=trust_remote_code)
        if hasattr(ref_config, "max_position_embeddings"):
            ref_config.max_position_embeddings = max(ref_config.max_position_embeddings, self.config.data.max_length)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not ref_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )
        with init_context():
            self.reference_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_reference_path,
                config=ref_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(
                    model=self.reference_model,
                    ulysses_sp_size=self.config.ulysses_sequence_parallel_size,
                )

        self.reference_model.requires_grad_(False)
        self.reference_model.eval()

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.reference_model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=False,
        )
        cpu_offload = None
        if self.config.model.fsdp_config.cpu_offload:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_reference_model = FSDP(
                self.reference_model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for FSDP2."
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.reference_model.state_dict()
            apply_fsdp2(self.reference_model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.reference_model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_reference_model = self.reference_model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        self.fsdp_reference_model.eval()
        log_gpu_memory_usage("After reference FSDP wrapping", logger=logger)

    def _sequence_logps(self, model, input_ids, attention_mask, position_ids, labels):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return get_batch_logps(outputs.logits, labels, average_log_prob=False)

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
            raise NotImplementedError("Offline DPO trainer currently requires use_remove_padding=False and SP=1.")

        chosen_input_ids = batch["chosen_input_ids"].to(self.device_name)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device_name)
        chosen_position_ids = batch["chosen_position_ids"].to(self.device_name)
        chosen_labels = batch["chosen_labels"].to(self.device_name)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device_name)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device_name)
        rejected_position_ids = batch["rejected_position_ids"].to(self.device_name)
        rejected_labels = batch["rejected_labels"].to(self.device_name)

        beta = self.config.algorithm.get("beta", 0.1)
        label_smoothing = self.config.algorithm.get("label_smoothing", 0.0)
        loss_type = self.config.algorithm.get("loss_type", "sigmoid")
        reference_free = self.config.algorithm.get("reference_free", False)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            policy_chosen_logps = self._sequence_logps(
                self.fsdp_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_position_ids,
                chosen_labels,
            )
            policy_rejected_logps = self._sequence_logps(
                self.fsdp_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_position_ids,
                rejected_labels,
            )
            with torch.no_grad():
                reference_chosen_logps = self._sequence_logps(
                    self.fsdp_reference_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_position_ids,
                    chosen_labels,
                )
                reference_rejected_logps = self._sequence_logps(
                    self.fsdp_reference_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_position_ids,
                    rejected_labels,
                )

            loss, dpo_logits = compute_dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=reference_chosen_logps,
                reference_rejected_logps=reference_rejected_logps,
                beta=beta,
                label_smoothing=label_smoothing,
                loss_type=loss_type,
                reference_free=reference_free,
            )
            loss = loss / n_micro_batches

        if do_backward:
            loss.backward()

        return loss, dpo_logits.detach().mean()

    def training_step(self, batch: TensorDict):
        start_time = time.time()
        self.fsdp_model.train()
        self.fsdp_reference_model.eval()
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0.0
        step_margin = 0.0
        for micro_batch in micro_batches:
            loss, margin = self._compute_loss_and_backward(batch=micro_batch, n_micro_batches=n_micro_batches)
            step_loss += loss.item()
            step_margin += margin.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()

        lr = self.lr_scheduler.get_last_lr()[0]
        step_loss = torch.tensor(step_loss, device=self.device_name)
        step_margin = torch.tensor(step_margin / max(n_micro_batches, 1), device=self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(step_margin, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            torch.distributed.all_reduce(step_margin)
            step_loss /= self.device_mesh.size(0)
            step_margin /= self.device_mesh.size(0)

        return {
            "train/loss": step_loss.detach().item(),
            "train/dpo_margin": step_margin.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": time.time() - start_time,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        self.fsdp_reference_model.eval()
        with torch.no_grad():
            loss, margin = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(margin, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                torch.distributed.all_reduce(margin)
                loss /= self.device_mesh.size(0)
                margin /= self.device_mesh.size(0)
        return loss, margin


def create_dpo_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    return OfflineDPODataset(
        parquet_files=data_paths,
        tokenizer=tokenizer,
        config=data_config,
        max_samples=max_samples,
    )


def run_dpo(config):
    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()
    from torch.distributed.device_mesh import init_device_mesh
    from verl.utils import hf_tokenizer

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    train_dataset = create_dpo_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_dpo_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        max_samples=config.data.get("val_max_samples", -1),
    )

    trainer = FSDPDPOTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    run_dpo(config)


if __name__ == "__main__":
    main()
