import torch
import torch.nn.functional as F


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Return sequence log-probabilities for labels, ignoring ``-100`` tokens."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels must have the same shape before the vocab dimension")

    labels = labels.contiguous().to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_logps = per_token_logps.view(shift_logits.size(0), shift_logits.size(1))

    loss_mask = shift_labels != -100
    sequence_logps = (per_token_logps * loss_mask).sum(dim=-1)

    if average_log_prob:
        num_valid_tokens = loss_mask.sum(dim=-1)
        return sequence_logps / torch.clamp(num_valid_tokens, min=1)
    return sequence_logps


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute offline DPO loss and return ``(loss, logits)``."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)

    logits = pi_logratios - ref_logratios
    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(
            -beta * logits
        ) * label_smoothing
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unsupported DPO loss_type: {loss_type}. Choose 'sigmoid' or 'ipo'.")

    return losses.mean(), logits
