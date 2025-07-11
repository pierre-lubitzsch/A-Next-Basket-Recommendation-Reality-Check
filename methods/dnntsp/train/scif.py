import math, random
from typing import List, Sequence, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import tqdm

from utils.data_container import get_data_loader, get_data_loader_temporal_split
from utils.util import save_model, convert_to_gpu, convert_all_data_to_gpu


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


def _batch_grad(model, data_loader, param_list,
                loss_func, average=True, average_scale=None):
    average_scale = average_scale or len(data_loader) * data_loader.batch_size
    acc = [torch.zeros_like(p) for p in param_list]

    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(data_loader):
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
            convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

        # (B, N)
        output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
        loss = loss_func(output, truth_data.float())
        
        grads = torch.autograd.grad(loss, param_list, retain_graph=False)
        for a, g in zip(acc, grads):
            a += g.detach()
    if average:
        for a in acc:
            a /= average_scale
    return acc


def _hvp_single(model, batch_tuple, v_list, param_list, loss_func):

    # with torch.backends.cudnn.flags(enabled=False):
    g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = batch_tuple
    g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
        convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

    # (B, N)
    output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
    loss = loss_func(output, truth_data.float())
    grad = torch.autograd.grad(loss, param_list, create_graph=True)
    dot  = sum((g * v).sum() for g, v in zip(grad, v_list))
    hv   = torch.autograd.grad(dot, param_list, retain_graph=False)
    return [h.detach() for h in hv]


def _hvp_dataset(
    model, batch,
    v_list, param_list,
    loss_func,
    average=True,
):
    # acc = [torch.zeros_like(p) for p in v_list]

    # for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(zip(*batch)):
    #     g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
    #         convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

    #     hv = _hvp_single(
    #         model, batch,
    #         v_list, param_list, loss_func,
    #     )
    #     for a, h in zip(acc, hv):
    #         a += h

    # if average:
    #     bs = len(batch) # maybe need to multiply by data_loader.batch_size
    #     for a in acc:
    #         a /= bs
    # return acc

    hv = _hvp_single(
        model, batch,
        v_list, param_list, loss_func,
    )

    if average:
        # batch[-1] is users_frequency, shape (batch_size, items_total)
        batch_size = batch[-1].shape[0]
        hv = [h / batch_size for h in hv]

    return hv

    

# sets2sets
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

# dnntsp
def target_params(model):
    params = []

    for n, p in model.named_parameters():
        if n == "item_embedding.weight":
            params.append(p)
        else:
            p.requires_grad_(False)
    
    return params


def scif_unlearn(
    *,  # force kw-only for clarity
    unlearning_user_ids: List[str],
    retain_user_ids: List[str],
    n: int,
    model: nn.Module,
    criterion: nn.Module,
    LOCAL: bool,
    temporal_split: bool,
    damping=0.01,
    scale=25.0,
    lissa_bs=16,
    retain_samples_used_for_update=128,
    train_pair_count=1024,
    history_path=None,
    future_path=None,
    user_to_unlearning_items=None,
):
    # 1) Make all BN layers use running stats (no per-batch norm)
    #    but keep the rest of the model in train() so grads flow.
    model.train()  
    for m in model.modules():
        # covers BatchNorm1d, 2d, 3d
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    # now get our list of parameters to unlearn
    param_list = target_params(model)

    # delete_pairs = [make_pair(u, True) for u in unlearning_user_ids]
    unlearn_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=1,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=False,
                                            unlearn_set_flag=True,
                                            users_in_unlearning_set=unlearning_user_ids,
                                            user_to_unlearning_items=user_to_unlearning_items,)

    batch_size = retain_samples_used_for_update + len(unlearning_user_ids)

    neg_grads = _batch_grad(model, unlearn_data_loader, param_list,
                            criterion, average_scale=batch_size)
    
    # unlearn this
    neg_grads = [-g for g in neg_grads]


    # need to sample retain pairs, otherwise it takes much too long
    train_retain_pair_count = train_pair_count - len(unlearning_user_ids)
    train_retain_pair_ids = random.sample(retain_user_ids, k=train_retain_pair_count)
    train_ids = unlearning_user_ids + train_retain_pair_ids

    k = max(0, retain_samples_used_for_update - len(unlearning_user_ids))
    more_retain_samples_needed = random.sample(retain_user_ids, k=k) if k > 0 else []
    retain_ids_sampled = unlearning_user_ids + more_retain_samples_needed


    retain_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=16,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=True,
                                            unlearn_set_flag=False,
                                            users_in_unlearning_set=unlearning_user_ids,
                                            user_to_unlearning_items=user_to_unlearning_items,
                                            user_subset=retain_ids_sampled,)

    pos_grads = _batch_grad(model, retain_data_loader, param_list,
                            criterion, average_scale=batch_size)

    grads = [n + p for n, p in zip(neg_grads, pos_grads)]

    # inv_hvp, _ = lissa_inv_hvp(encoder, decoder, train_pairs, grads, param_list,
    #                            codes_inverse_freq, criterion, output_size, max_len,
    #                            damping=damping, scale=scale, bs=lissa_bs, LOCAL=LOCAL)

    cur_train_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=lissa_bs,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=True,
                                            unlearn_set_flag=False,
                                            users_in_unlearning_set=unlearning_user_ids,
                                            user_to_unlearning_items=user_to_unlearning_items,
                                            user_subset=train_ids,
                                            shuffle=True,)


    inv_hvp, _ = cg_inv_hvp(
        model, cur_train_data_loader, grads, param_list,
        criterion,
        damping=damping,           # reuse existing kwargs
        bs=lissa_bs,               # same mini‑batch size
        LOCAL=LOCAL,
    )


    tau = 1 / n
    with torch.no_grad():
        for p, d in zip(param_list, inv_hvp):
            p -= tau * d

    print(f"[SCIF]  removed {len(unlearning_user_ids)} baskets, "
          f"tau={tau},  ||delta theta||={tau * norm_list(inv_hvp)}")



# conjugate gradients:

###############################################################################
#           Conjugate‑gradients inverse‑Hessian‑vector product                #
###############################################################################

def _dot_list(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    """⟨a,b⟩ for lists of tensors (returned as a scalar tensor on the same device)."""
    return sum((x * y).sum() for x, y in zip(a, b))


def _add_scaled(x: Sequence[torch.Tensor],
                y: Sequence[torch.Tensor],
                alpha: float) -> List[torch.Tensor]:
    """x + α·y (list form, no gradients kept)."""
    return [xi + alpha * yi for xi, yi in zip(x, y)]


def cg_inv_hvp(
    model,                             # models
    train_data_loader,
    v_list: Sequence[torch.Tensor],               # right‑hand side ‘v’
    param_list: Sequence[torch.Tensor],           # θ we differentiate w.r.t.
    criterion,
    damping: float = 0.01,                        # λ – Tikhonov damping
    bs: int = 16,                                 # mini‑batch size for H·p
    max_iter: int | None = None,
    tol: float = 1e-5,
    LOCAL: bool = False,
):
    """
    Solve  (H + λI) x = v  for x with conjugate gradients.
    Returns: (x, diverged_flag)
    """
    # Total number of iterations: one pass over the data by default
    max_iter = max_iter or math.ceil(len(train_data_loader) / bs)

    # --- initialisation ------------------------------------------------------
    x      = [torch.zeros_like(v) for v in v_list]   # x₀
    r      = [v.clone() for v in v_list]             # r₀ = v − Hx₀  (Hx₀ = 0)
    p      = [ri.clone() for ri in r]                # p₀ = r₀
    rs_old = _dot_list(r, r).item()

    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in tqdm.tqdm(enumerate(train_data_loader), disable=not LOCAL):
        if step >= max_iter:
            break

        q = _hvp_dataset(
            model, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency),
            p, param_list,
            criterion,
            average=True,
        )
        # add λI term
        q = [qi + damping * pi for qi, pi in zip(q, p)]

        # ---------------------------------------------------------------------
        alpha = rs_old / _dot_list(p, q).item()

        # x_{k+1}  =  x_k + α p_k
        x = _add_scaled(x, p, alpha)

        # r_{k+1}  =  r_k − α q
        r = _add_scaled(r, q, -alpha)

        rs_new = _dot_list(r, r).item()
        if math.sqrt(rs_new) < tol:
            break  # converged

        beta = rs_new / rs_old

        # p_{k+1}  =  r_{k+1} + β p_k
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rs_old = rs_new

    diverged = math.sqrt(rs_old) >= tol
    return x, diverged
