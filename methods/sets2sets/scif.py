import math, random
from typing import List, Sequence, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import tqdm


def list_to_vec(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])

def vec_to_list(vec: torch.Tensor, like: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    out, idx = [], 0
    for p in like:
        n = p.numel()
        out.append(vec[idx:idx+n].view_as(p))
        idx += n
    return out

def norm_list(plist: Sequence[torch.Tensor]) -> float:
    return torch.sqrt(sum(p.pow(2).sum() for p in plist)).item()

def _loss_forward(
    encoder, decoder, input_var, target_var,
    codes_inverse_freq, criterion, output_size, max_len
):
    """
    Exact same computation as `train()` - *minus* optim.step() and *minus*
    gradient zeroing.  Returns the scalar loss so we can auto-diff it.
    """
    use_cuda = next(encoder.parameters()).is_cuda
    encoder_hidden = encoder.initHidden()
    input_len = len(input_var)

    # shape: [MAX_LENGTH, hidden_size]
    encoder_outputs = Variable(
        torch.zeros(max_len, encoder.hidden_size,
                    device="cuda" if use_cuda else "cpu")
    )

    # history frequency vector
    hist = np.zeros(output_size)
    for ei in range(1, input_len - 1):
        for ele in input_var[ei]:
            hist[ele] += 1.0 / (input_len - 2)

    for ei in range(1, input_len - 1):
        enc_out, encoder_hidden = encoder(input_var[ei], encoder_hidden)
        encoder_outputs[ei - 1] = enc_out[0][0]

    last_input = input_var[input_len - 2]
    decoder_hidden = encoder_hidden
    decoder_input  = last_input

    decoder_out, _, _ = decoder(
        decoder_input, decoder_hidden, encoder_outputs, hist, encoder_hidden
    )

    # vectorise target basket
    vec_tgt = np.zeros(output_size)
    for idx in target_var[1]:
        vec_tgt[idx] = 1
    tgt = torch.as_tensor(vec_tgt, dtype=torch.float32,
                          device="cuda" if use_cuda else "cpu").view(1, -1)

    weights = torch.as_tensor(
        codes_inverse_freq, dtype=torch.float32,
        device="cuda" if use_cuda else "cpu").view(1, -1)

    return criterion(decoder_out, tgt, weights)


def _batch_grad(encoder, decoder, batch, param_list,
                codes_inverse_freq, criterion, output_size, max_len, average=True, average_scale=None):
    average_scale = average_scale or len(batch)
    acc = [torch.zeros_like(p) for p in param_list]
    for inp, tgt in batch:
        loss = _loss_forward(encoder, decoder, inp, tgt,
                             codes_inverse_freq, criterion, output_size, max_len)
        grads = torch.autograd.grad(loss, param_list, retain_graph=False)
        for a, g in zip(acc, grads):
            a += g.detach()
    if average:
        for a in acc:
            a /= average_scale
    return acc

def _hvp_single(encoder, decoder, inp, tgt, v_list, param_list,
                codes_inverse_freq, criterion, output_size, max_len):
    # second derivative not supported for RNNs when using cuDNN...
    with torch.backends.cudnn.flags(enabled=False):
        loss = _loss_forward(encoder, decoder, inp, tgt,
                         codes_inverse_freq, criterion, output_size, max_len)
        grad = torch.autograd.grad(loss, param_list, create_graph=True)
        dot  = sum((g * v).sum() for g, v in zip(grad, v_list))
        hv   = torch.autograd.grad(dot, param_list, retain_graph=False)
    return [h.detach() for h in hv]


def _hvp_dataset(
    encoder, decoder, data_batch,
    v_list, param_list,
    codes_inverse_freq, criterion, output_size, max_len, average=True,
):
    acc = [torch.zeros_like(p) for p in v_list]
    for inp, tgt in data_batch:
        hv = _hvp_single(
            encoder, decoder, inp, tgt,
            v_list, param_list,
            codes_inverse_freq, criterion, output_size, max_len
        )
        for a, h in zip(acc, hv):
            a += h
    if average:
        bs = len(data_batch)
        for a in acc:
            a /= bs
    return acc

def lissa_inv_hvp(
    encoder, decoder, train_data, v_list, param_list,
    codes_inverse_freq, criterion, output_size, max_len,
    damping=0.01, scale=25., bs=16, max_iter=None, tol=1e-5, LOCAL=False,
):
    """
    Vector-Jacobian recursion from Agarwal et al.

    Returns H⁻¹v (list-form) and a bool diverged flag.
    """
    max_iter = max_iter or math.ceil(len(train_data) / bs)
    cur_est = [vi.clone() for vi in v_list]

    for it in tqdm.tqdm(range(max_iter), disable=not LOCAL):
        # mini-batch
        idx = [random.randrange(len(train_data)) for _ in range(bs)]
        batch = [train_data[i] for i in idx]

        hv = _hvp_dataset(
            encoder, decoder, batch, cur_est, param_list,
            codes_inverse_freq, criterion, output_size, max_len
        )

        # u_{t+1} = v + (1-damp)·u_t − (1/scale)·H·u_t
        next_est = [
            vi + (1 - damping) * ui - h / scale
            for vi, ui, h in zip(v_list, cur_est, hv)
        ]
        delta = norm_list([n - c for n, c in zip(next_est, cur_est)])
        if delta < tol:
            break
        cur_est = next_est

    return cur_est, False   # never mark diverged for simplicity


def target_params(encoder, decoder):
    enc, dec = [], []

    for n, p in encoder.named_parameters():
        if n == "embedding.weight":
            enc.append(p)                     # keep
        else:
            p.requires_grad_(False)           # freeze

    keep = {
        "embedding.weight",
        "out.weight", "out.bias",
        "attn_combine5.weight", "attn_combine5.bias",
    }
    for n, p in decoder.named_parameters():
        if n in keep:
            dec.append(p)
        else:
            p.requires_grad_(False)

    return enc + dec


def scif_unlearn(
    *,  # force kw-only for clarity
    unlearning_user_ids: List[str],
    retain_user_ids: List[str],
    cur_clean_data_history_and_future: Dict[str, list],
    history_data: Dict[str, list],
    future_data:  Dict[str, list],
    encoder: nn.Module,
    decoder: nn.Module,
    codes_inverse_freq: np.ndarray,
    criterion: nn.Module,
    output_size: int,
    LOCAL: bool,
    temporal_split: bool,
    retain_pairs=None,
    max_len=100,
    damping=0.01,
    scale=25.0,
    lissa_bs=16,
    retain_samples_used_for_update=128,
    train_pair_count=1024,
    max_norm=None,
):

    def make_pair(u, sensitive_included):
        if temporal_split:
            if sensitive_included:
                # training basket = last-3 in history
                tgt = [[-1], history_data[u][-3], [-1]]
                inp = history_data[u][:-3] + [[-1]]
            else:  # clean sample
                tgt = [[-1], cur_clean_data_history_and_future[u][-4], [-1]]
                inp = cur_clean_data_history_and_future[u][:-4] + [[-1]]
        else:
            inp = history_data[u]
            tgt = future_data[u]
        return inp, tgt

    param_list = target_params(encoder, decoder) 

    delete_pairs = [make_pair(u, True) for u in unlearning_user_ids]

    batch_size = retain_samples_used_for_update + len(unlearning_user_ids)

    neg_grads = _batch_grad(encoder, decoder, delete_pairs, param_list,
                            codes_inverse_freq, criterion, output_size, max_len, average_scale=batch_size)
    
    # unlearn this
    neg_grads = [-g for g in neg_grads]
    
    # learn this (retain set and modified sample without sensitive items)
    unlearn_samples_corrected = [make_pair(u, False) for u in unlearning_user_ids if u in cur_clean_data_history_and_future]

    # need to sample retain pairs, otherwise it takes much too long
    train_retain_pair_count = min(train_pair_count - len(unlearn_samples_corrected), len(retain_pairs))
    train_retain_pair_samples = random.sample(retain_pairs, k=train_retain_pair_count)
    train_pairs = unlearn_samples_corrected + train_retain_pair_samples

    k = max(0, retain_samples_used_for_update - len(unlearn_samples_corrected))
    more_retain_samples_needed = random.sample(retain_pairs, k=k) if k > 0 else []
    retain_pairs_sampled = unlearn_samples_corrected + more_retain_samples_needed

    pos_grads = _batch_grad(encoder, decoder, retain_pairs_sampled, param_list,
                            codes_inverse_freq, criterion, output_size, max_len, average_scale=batch_size)

    grads = [n + p for n, p in zip(neg_grads, pos_grads)]

    # inv_hvp, _ = lissa_inv_hvp(encoder, decoder, train_pairs, grads, param_list,
    #                            codes_inverse_freq, criterion, output_size, max_len,
    #                            damping=damping, scale=scale, bs=lissa_bs, LOCAL=LOCAL)

    inv_hvp, _ = cg_inv_hvp(
        encoder, decoder, train_pairs, grads, param_list,
        codes_inverse_freq, criterion, output_size, max_len,
        damping=damping,           # reuse existing kwargs
        bs=lissa_bs,               # same mini‑batch size
        LOCAL=LOCAL,
    )

    # clip the norm of inv_hvp to max_norm
    tau = 1 / len(history_data)
    if max_norm is not None:
        delta_norm = tau * norm_list(inv_hvp)
        if delta_norm > max_norm:
            scale = max_norm / delta_norm
            inv_hvp = [x * scale for x in inv_hvp]

    with torch.no_grad():
        for p, d in zip(param_list, inv_hvp):
            p -= tau * d

    print(f"[SCIF]  removed {len(delete_pairs)} baskets, "
          f"tau={tau:.4f},  ||delta theta||={tau * norm_list(inv_hvp)}")



# conjugate gradients:

###############################################################################
#           Conjugate‑gradients inverse‑Hessian‑vector product                #
###############################################################################

def _dot_list(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    """dot product for lists of tensors (returned as a scalar tensor on the same device)."""
    return sum((x * y).sum() for x, y in zip(a, b))


def _add_scaled(x: Sequence[torch.Tensor],
                y: Sequence[torch.Tensor],
                alpha: float) -> List[torch.Tensor]:
    """x + alpha y (list form, no gradients kept)."""
    return [xi + alpha * yi for xi, yi in zip(x, y)]


def cg_inv_hvp(
    encoder, decoder,                             # models
    train_data: List[Tuple],                      # data used to build H
    v_list: Sequence[torch.Tensor],               # right‑hand side ‘v’
    param_list: Sequence[torch.Tensor],           # theta we differentiate w.r.t.
    codes_inverse_freq, criterion, output_size, max_len,
    damping: float = 0.01,                        # lambda – Tikhonov damping
    bs: int = 16,                                 # mini‑batch size for H·p
    max_iter: int | None = None,
    tol: float = 1e-5,
    LOCAL: bool = False,
):
    """
    Solve  (H + lambda I) x = v  for x with conjugate gradients.
    Returns: (x, diverged_flag)
    """
    # Total number of iterations: one pass over the data by default
    max_iter = max_iter or math.ceil(len(train_data) / bs)

    # --- initialisation ------------------------------------------------------
    x      = [torch.zeros_like(v) for v in v_list]   # x_theta
    r      = [v.clone() for v in v_list]             # r_theta = v − H x_theta  (H x_theta = 0)
    p      = [ri.clone() for ri in r]                # p_theta = r_theta
    rs_old = _dot_list(r, r).item()

    for k in tqdm.tqdm(range(max_iter), disable=not LOCAL):
        # ---------------------------------------------------------------------
        # Compute  q = (H + lambda I) p  using stochastic HVP on a mini‑batch
        # ---------------------------------------------------------------------
        idx   = [random.randrange(len(train_data)) for _ in range(bs)]
        batch = [train_data[i] for i in idx]

        q = _hvp_dataset(
            encoder, decoder, batch,
            p, param_list,
            codes_inverse_freq, criterion, output_size, max_len,
            average=True,
        )
        # add lambda I term
        q = [qi + damping * pi for qi, pi in zip(q, p)]

        # ---------------------------------------------------------------------
        if _dot_list(p, q).item() == 0:
            # p and q are orthogonal, we cannot proceed
            print(f"[CG]  p and q are orthogonal at iteration {k}, stopping.")
            break
        alpha = rs_old / _dot_list(p, q).item()

        # x_{k+1}  =  x_k + alpha p_k
        x = _add_scaled(x, p, alpha)

        # r_{k+1}  =  r_k − alpha q
        r = _add_scaled(r, q, -alpha)

        rs_new = _dot_list(r, r).item()
        if math.sqrt(rs_new) < tol:
            break  # converged

        beta = rs_new / rs_old

        # p_{k+1}  =  r_{k+1} + beta p_k
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new

    diverged = math.sqrt(rs_old) >= tol
    return x, diverged
