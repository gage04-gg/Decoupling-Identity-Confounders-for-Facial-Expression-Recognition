from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
def shuffle_batch(x: torch.Tensor) -> torch.Tensor:
    if x.size(0) < 2:
        raise ValueError("MI estimation requires batch size >= 2.")
    shift = int(torch.randint(1, x.size(0), (), device=x.device).item())
    return x.roll(shift, dims=0)


def dv_bound(
    joint_scores: torch.Tensor,
    marginal_scores: torch.Tensor,
    sum_spatial: bool = False,
) -> torch.Tensor:
    joint = joint_scores.flatten(start_dim=1)
    marginal = marginal_scores.flatten(start_dim=1)
    log_mean_exp_marginal = torch.logsumexp(marginal, dim=0) - torch.log(
        marginal.new_tensor(float(marginal.size(0)))
    )
    estimates = joint.mean(dim=0) - log_mean_exp_marginal

    return estimates.sum() if sum_spatial else estimates.mean()


def estimate_global_mi(
    statistics_net: nn.Module,
    image_features: torch.Tensor,
    representation: torch.Tensor,
) -> torch.Tensor:
    joint = statistics_net(image_features, representation)
    marginal = statistics_net(image_features, shuffle_batch(representation))
    return dv_bound(joint, marginal, sum_spatial=False)


def estimate_local_mi(
    statistics_net: nn.Module,
    local_features: torch.Tensor,
    representation: torch.Tensor,
) -> torch.Tensor:
    joint = statistics_net(local_features, representation)
    marginal = statistics_net(local_features, shuffle_batch(representation))

    return dv_bound(joint, marginal, sum_spatial=True)


def discriminator_loss(
    discriminator: nn.Module,
    expression: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    joint_logits = discriminator(expression.detach(), identity.detach())
    marginal_logits = discriminator(
        expression.detach(),
        shuffle_batch(identity.detach()),
    )

    joint_loss = F.binary_cross_entropy_with_logits(
        joint_logits,
        torch.zeros_like(joint_logits),
    )
    marginal_loss = F.binary_cross_entropy_with_logits(
        marginal_logits,
        torch.ones_like(marginal_logits),
    )

    return joint_loss + marginal_loss


def encoder_adversarial_loss(
    discriminator: nn.Module,
    expression: torch.Tensor,
    identity: torch.Tensor,
) -> torch.Tensor:
    joint_logits = discriminator(expression, identity)

    return F.binary_cross_entropy_with_logits(
        joint_logits,
        torch.ones_like(joint_logits),
    )
    
