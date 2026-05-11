import os
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_sample(loss):
    return loss.mean(dim=(1, 2, 3)) if len(loss.shape) > 1 else loss


def task_loss(logits, target, reduction="mean"):
    loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
    if not isinstance(logits, (list, tuple)):
        loss = loss_fn(logits, target)
    else:
        loss = sum(loss_fn(logit, target) for logit in logits)
        if reduction == "mean":
            loss = loss / len(logits)
    if reduction == "none" and len(loss.shape) > 1:
        loss = loss.mean(dim=(1, 2, 3))
    return loss


def kd_loss(student_logits, teacher_logits, reduction="mean", T=4.0):
    if isinstance(teacher_logits, (list, tuple)):
        teacher_logits = teacher_logits[0]
    teacher_prob = torch.sigmoid(teacher_logits / T)

    if isinstance(student_logits, (list, tuple)):
        total_loss = 0.0
        for student_logit in student_logits:
            student_prob = torch.sigmoid(student_logit / T)
            teacher_prob_cur = teacher_prob
            if teacher_prob_cur.shape[-2:] != student_prob.shape[-2:]:
                teacher_prob_cur = F.interpolate(
                    teacher_prob_cur,
                    size=student_prob.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss = F.mse_loss(student_prob, teacher_prob_cur, reduction=reduction)
            total_loss += _per_sample(loss) if reduction == "none" else loss
        if reduction == "mean":
            return total_loss / len(student_logits)
        if reduction == "none":
            return total_loss / len(student_logits)
        return total_loss

    student_prob = torch.sigmoid(student_logits / T)
    loss = F.mse_loss(student_prob, teacher_prob, reduction=reduction)
    return _per_sample(loss) if reduction == "none" else loss


def gate_l1_regularizer(teacher_model):
    return teacher_model.modulator.gamma_gate.abs().mean()


def _cluster_weights(alpha, img_id, cluster_map, device, detach_alpha):
    cluster_ids = torch.tensor(
        [int(cluster_map[os.path.basename(name + ".png")]) for name in img_id],
        dtype=torch.long,
        device=device,
    )
    raw = alpha[cluster_ids]
    if detach_alpha:
        raw = raw.detach()
    raw = raw - raw.mean()
    raw = log(2.0) * torch.tanh(raw / log(2.0))
    weights = torch.exp(raw)
    return weights / (weights.mean().detach() + 1e-6)


def outer_loss(student, teacher, alpha, batch, cluster_map, device):
    img_cnn, mask, _, img_dino, img_id = batch
    img_cnn = img_cnn.to(device)
    mask = mask.unsqueeze(1).to(device)
    img_dino = img_dino.to(device)
    weights = _cluster_weights(alpha, img_id, cluster_map, device, detach_alpha=False)

    with torch.no_grad():
        teacher_logits = teacher(img_cnn=img_cnn, img_dino=img_dino)
    student_logits = student(img_cnn)

    loss_kd = kd_loss(student_logits, teacher_logits, reduction="none", T=4.0)
    loss_task = task_loss(student_logits, mask, reduction="none")
    return (weights * (loss_kd + loss_task)).mean()


def inner_loss(student, teacher, alpha, data, targets, image_dino, img_id, cluster_map, device):
    teacher_logits = teacher(img_cnn=data, img_dino=image_dino)
    with torch.no_grad():
        student_logits = student(data)

    loss_t_task = task_loss(teacher_logits, targets, reduction="none")
    loss_align = kd_loss(student_logits, teacher_logits, reduction="none", T=4.0)
    loss_gate = gate_l1_regularizer(teacher)
    weights = _cluster_weights(alpha, img_id, cluster_map, device, detach_alpha=True)

    return (weights * (loss_t_task + 0.1 * loss_align)).mean() + 0.005 * loss_gate, teacher_logits
