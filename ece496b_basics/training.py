import collections
import math
import os
import regex as re
from typing import IO, BinaryIO, Iterable, Optional, Type

import numpy.typing as npt
import torch


def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    max_vals = torch.max(inputs, dim=1, keepdim=True)[0]
    inputs = inputs - max_vals
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs), dim=1, keepdim=True))
    log_probs = inputs - log_sum_exp
    loss = -log_probs[torch.arange(targets.shape[0]), targets]

    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Implements AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate (default: 1e-3).
            betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
            eps: Term added to denominator to improve numerical stability (default: 1e-8).
            weight_decay: Weight decay coefficient (default: 1e-2).
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(param.data)
                    state['v'] = torch.zeros_like(param.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                state['step'] += 1
                m.mul_(beta1).add_(grad, alpha=1 - beta1)  # m <- B1m + (1 - B1)g
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v <- B2v + (1 - B2)g^2

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                param.data.addcdiv_(m, v.sqrt().add_(group['eps']), value=-step_size)
                
                if group['weight_decay'] != 0:
                    param.data.add_(param.data, alpha=-group['lr'] * group['weight_decay'])

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters: int
            T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        )
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters: collection of trainable parameters.
        max_l2_norm: a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.

    Returns:
        None
    """
    epsilon = 1e-6
    # Flatten and concatenate all gradients into a single tensor
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads))

    if total_norm > max_l2_norm:
        scale_factor = max_l2_norm / (total_norm + epsilon)
        for g in grads:
            g.mul_(scale_factor)