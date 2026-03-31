"""
losses.py  —  All loss functions for the hypernetwork training pipeline.

Three losses are combined:

  L_total = λ_task · L_task  +  λ_recon · L_recon  +  λ_distill · L_distill

─────────────────────────────────────────────────────────────────────────────
Loss 1: Task Loss  (L_task)
───────────────────────────
  Standard token-level cross-entropy over the language modelling objective.

  L_task = CrossEntropy(logits_student, targets)

  PyTorch:
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        labels[:, 1:].reshape(-1),
        ignore_index=-1
    )

─────────────────────────────────────────────────────────────────────────────
Loss 2: Weight Reconstruction Loss  (L_recon)
──────────────────────────────────────────────
  MSE between hypernetwork-generated weights and frozen teacher weights.

  L_recon = (1/N) Σ_layers Σ_keys  ‖W_gen - W_teacher‖²_F / (out × in)

  Normalised by the matrix size so all weight shapes contribute equally.

  PyTorch:
    loss = mean(  (W_gen - W_teacher)**2  )   per weight, then averaged

─────────────────────────────────────────────────────────────────────────────
Loss 3: Knowledge Distillation Loss  (L_distill)
─────────────────────────────────────────────────
  KL divergence between softened teacher and student output distributions.
  Hinton et al. (2015) formulation.

  Soft targets:
    p_teacher = softmax(logits_teacher / T)
    p_student = softmax(logits_student / T)

  L_distill = T² · KL(p_teacher ‖ p_student)
            = T² · Σ p_teacher · (log p_teacher − log p_student)

  Note: multiply by T² to keep gradient scale consistent across temperatures.

  PyTorch:
    loss = T**2 * F.kl_div(
        F.log_softmax(logits_s / T, dim=-1),
        F.softmax(logits_t / T, dim=-1),
        reduction='batchmean'
    )

─────────────────────────────────────────────────────────────────────────────
Optional: Hidden-State Distillation  (L_hidden)
───────────────────────────────────────────────
  MSE between intermediate hidden states (layer-wise).

  L_hidden = (1/L) Σ_l  ‖h_student_l - h_teacher_l‖² / D

─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainingConfig


# ---------------------------------------------------------------------------
# Loss 1: Task loss
# ---------------------------------------------------------------------------

def task_loss(
    logits: torch.Tensor,    # [B, S, V]
    labels: torch.Tensor,    # [B, S]
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Next-token prediction cross-entropy.

    Shifts inputs left by 1 so that position t predicts token t+1.

    Returns scalar tensor.
    """
    # Shift: logits [B, S-1, V],  labels [B, S-1]
    shift_logits = logits[:, :-1, :].contiguous()    # [B, S-1, V]
    shift_labels = labels[:, 1:].contiguous()         # [B, S-1]

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


# ---------------------------------------------------------------------------
# Loss 2: Weight reconstruction loss
# ---------------------------------------------------------------------------

def weight_reconstruction_loss(
    generated_weights: List[Dict[str, torch.Tensor]],
    teacher_weights:   List[Dict[str, torch.Tensor]],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Mean squared error between generated and teacher weights, normalised
    per-matrix by its number of elements.

    Parameters
    ----------
    generated_weights : list of {key: tensor} per layer
    teacher_weights   : list of {key: tensor} per layer  (detached)
    normalize         : whether to divide each term by element count

    Returns scalar tensor.
    """
    total_loss  = torch.tensor(0.0, requires_grad=True)
    total_count = 0

    for gen_layer, tea_layer in zip(generated_weights, teacher_weights):
        for key in gen_layer:
            if key not in tea_layer:
                continue
            W_gen = gen_layer[key]
            W_tea = tea_layer[key].detach()

            if W_gen.shape != W_tea.shape:
                # Shouldn't happen — safety guard
                continue

            if normalize:
                n    = W_gen.numel()
                term = ((W_gen - W_tea) ** 2).sum() / n
            else:
                term = F.mse_loss(W_gen, W_tea, reduction="mean")

            total_loss  = total_loss + term
            total_count += 1

    if total_count == 0:
        return torch.tensor(0.0)

    return total_loss / total_count


# ---------------------------------------------------------------------------
# Loss 3: Knowledge distillation loss (Hinton 2015)
# ---------------------------------------------------------------------------

def distillation_loss(
    logits_student: torch.Tensor,   # [B, S, V]
    logits_teacher: torch.Tensor,   # [B, S, V]  (detached from teacher model)
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    KL-divergence distillation loss with temperature scaling.

    L = T² · KL(p_teacher ‖ p_student)

    where p = softmax(logits / T).

    Returns scalar tensor.
    """
    T = temperature
    B, S, V = logits_student.shape

    # Flatten to [B*S, V]
    ls = logits_student.view(-1, V) / T
    lt = logits_teacher.detach().view(-1, V) / T

    # log-softmax for student, softmax for teacher
    log_p_s = F.log_softmax(ls, dim=-1)    # [B*S, V]
    p_t     = F.softmax(lt,    dim=-1)     # [B*S, V]

    # KL(p_teacher ‖ p_student) = Σ p_t * (log p_t - log p_s)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean")

    return (T ** 2) * kl


# ---------------------------------------------------------------------------
# Loss 4: Hidden-state distillation (optional)
# ---------------------------------------------------------------------------

def hidden_state_distillation_loss(
    hidden_student: List[torch.Tensor],   # [L] each [B, S, D]
    hidden_teacher: List[torch.Tensor],   # [L] each [B, S, D]
) -> torch.Tensor:
    """
    MSE between per-layer hidden states, normalised by hidden dimension.

    Returns scalar tensor.
    """
    assert len(hidden_student) == len(hidden_teacher)
    total = torch.tensor(0.0, requires_grad=True)

    for h_s, h_t in zip(hidden_student, hidden_teacher):
        D    = h_s.size(-1)
        term = ((h_s - h_t.detach()) ** 2).mean() / D
        total = total + term

    return total / len(hidden_student)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class HypernetworkLoss(nn.Module):
    """
    Combines all losses with configurable weights.

    Example forward call:
        loss_dict = criterion(
            logits_student,
            logits_teacher,
            labels,
            generated_weights,
            teacher_weights,
        )
        loss_dict["total"].backward()
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.lambda_task    = cfg.lambda_task
        self.lambda_recon   = cfg.lambda_recon
        self.lambda_distill = cfg.lambda_distill
        self.temperature    = cfg.temperature

    def forward(
        self,
        logits_student:    torch.Tensor,                   # [B, S, V]
        logits_teacher:    torch.Tensor,                   # [B, S, V]
        labels:            torch.Tensor,                   # [B, S]
        generated_weights: List[Dict[str, torch.Tensor]],
        teacher_weights:   List[Dict[str, torch.Tensor]],
        hidden_student:    Optional[List[torch.Tensor]] = None,
        hidden_teacher:    Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with keys:
            'task'    — cross-entropy loss
            'recon'   — weight reconstruction loss
            'distill' — knowledge distillation loss
            'hidden'  — hidden-state distillation (if hiddens provided)
            'total'   — weighted sum
        """
        L_task    = task_loss(logits_student, labels)
        L_recon   = weight_reconstruction_loss(generated_weights, teacher_weights)
        L_distill = distillation_loss(logits_student, logits_teacher, self.temperature)

        total = (self.lambda_task    * L_task +
                 self.lambda_recon   * L_recon +
                 self.lambda_distill * L_distill)

        result = {
            "task":    L_task,
            "recon":   L_recon,
            "distill": L_distill,
            "total":   total,
        }

        if hidden_student is not None and hidden_teacher is not None:
            L_hidden = hidden_state_distillation_loss(hidden_student, hidden_teacher)
            result["hidden"] = L_hidden
            result["total"]  = total + 0.1 * L_hidden  # fixed small weight

        return result


# ---------------------------------------------------------------------------
# Perplexity (evaluation metric)
# ---------------------------------------------------------------------------

def compute_perplexity(
    model:      nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device:     torch.device,
    generated_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    max_batches: int = 100,
) -> float:
    """
    Compute perplexity = exp(mean_cross_entropy).

    Works for both standalone and generated-weights modes.
    """
    model.eval()
    total_loss    = 0.0
    total_tokens  = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch.get("labels", input_ids).to(device)

            logits, loss = model(input_ids, generated_weights, labels)
            if loss is None:
                loss = task_loss(logits, labels)

            # Count non-padding tokens
            valid_tokens  = (labels[:, 1:] != -1).sum().item()
            total_loss   += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(avg_loss)))


# ---------------------------------------------------------------------------
# Reconstruction error summary
# ---------------------------------------------------------------------------

def weight_reconstruction_report(
    generated_weights: List[Dict[str, torch.Tensor]],
    teacher_weights:   List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """
    Per-key MSE and relative error (‖W_gen - W_tea‖_F / ‖W_tea‖_F).

    Useful for diagnosing which weight types are hardest to reconstruct.
    """
    report: Dict[str, List[float]] = {}

    for gen_layer, tea_layer in zip(generated_weights, teacher_weights):
        for key in gen_layer:
            if key not in tea_layer:
                continue
            W_g = gen_layer[key].detach().float()
            W_t = tea_layer[key].detach().float()
            mse  = ((W_g - W_t) ** 2).mean().item()
            rel  = (W_g - W_t).norm().item() / (W_t.norm().item() + 1e-8)
            if key not in report:
                report[key] = []
            report[key].append((mse, rel))

    summary: Dict[str, float] = {}
    for key, vals in report.items():
        mses, rels = zip(*vals)
        summary[f"{key}_mse_avg"] = sum(mses) / len(mses)
        summary[f"{key}_rel_avg"] = sum(rels) / len(rels)

    return summary
