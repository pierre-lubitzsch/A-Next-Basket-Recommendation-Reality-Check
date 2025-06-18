import math, random
from typing import List, Sequence, Tuple, Dict
import torch, torch.nn as nn, torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import tqdm
import sys
import tqdm
import json


from utils.data_container import get_data_loader, get_data_loader_temporal_split
from utils.util import save_model, convert_to_gpu, convert_all_data_to_gpu
from utils.metric import get_metric


# def _batch_grad_pairs(pairs, encoder, decoder, param_list,
#                       codes_inverse_freq, criterion,
#                       output_size, max_len, device) -> List[torch.Tensor]:
#     """Average gradient over a list of (inp,tgt) pairs."""
#     acc = [torch.zeros_like(p, device=device) for p in param_list]
#     for inp, tgt in pairs:
#         loss = _loss_forward(encoder, decoder, inp, tgt,
#                              codes_inverse_freq, criterion,
#                              output_size, max_len)
#         grads = torch.autograd.grad(
#             loss,
#             param_list,
#             retain_graph=False,
#             allow_unused=True
#         )
#         # fill in zeros for any None
#         grads = [
#             g if g is not None else torch.zeros_like(p)
#             for p, g in zip(param_list, grads)
#         ]
#         for a, g in zip(acc, grads):
#             a += g.detach()
#     for a in acc:                        # mean
#         a /= max(1, len(pairs))
#     return acc


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


def _mean_abs(t: torch.Tensor) -> torch.Tensor:
    return t.abs().mean()


def _reset_adam_state(opt: torch.optim.Adam, params: List[torch.Tensor]):
    for p in params:
        if p in opt.state:
            opt.state[p]['step'] = torch.zeros(1, dtype=torch.float32, device=p.device)
            opt.state[p]['exp_avg'].zero_()
            opt.state[p]['exp_avg_sq'].zero_()


# unlearning_user_ids: List[str],
#     retain_user_ids: List[str],
#     n: int,
#     model: nn.Module,
#     criterion: nn.Module,
#     LOCAL: bool,
#     temporal_split: bool,
#     damping=0.01,
#     scale=25.0,
#     lissa_bs=16,
#     retain_samples_used_for_update=128,
#     train_pair_count=1024,
#     history_path=None,
#     future_path=None,
#     user_to_unlearning_items=None,

def unlearn_by_reinit_and_finetune(
    *,
    unlearning_user_ids: List[str],
    retain_user_ids: List[str],
    model: nn.Module,
    criterion: nn.Module,
    LOCAL: bool,
    temporal_split: bool = True,
    kookmin_init_rate: float = 0.01,     # 1 % of lowest-|grad| params
    device: str = "cuda",
    neg_grad_retain_sample_size: int = 128,
    retain_epoch_count=5,
    optimizer=None,
    param_list=None,
    param_index=None,
    retain_samples_used_for_update=32,
    history_path=None,
    future_path=None,
    user_to_unlearning_items=None,
    trainable_cleaned_unlearn_user_ids=None,
):
    """
    Re-implements Kookmin’s low-|grad| re-init but with *encoder/decoder*
    instead of a monolithic model.
    """
    # 1) Make all BN layers use running stats (no per-batch norm)
    #    but keep the rest of the model in train() so grads flow.
    model.train()  
    for m in model.modules():
        # covers BatchNorm1d, 2d, 3d
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


    unlearn_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=1,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=False,
                                            unlearn_set_flag=True,
                                            users_in_unlearning_set=unlearning_user_ids,
                                            user_to_unlearning_items=user_to_unlearning_items,)

    # sample additional retain pairs so that we have
    # `neg_grad_retain_sample_size` in total
    k_more = max(0, neg_grad_retain_sample_size - len(trainable_cleaned_unlearn_user_ids))
    extra_retain = random.sample(retain_user_ids, k=k_more) if k_more else []
    retain_ids_sampled = trainable_cleaned_unlearn_user_ids + extra_retain

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

    

    # grads_forget = _batch_grad_pairs(
    #     forget_pairs, encoder, decoder, param_list,
    #     codes_inverse_freq, criterion, output_size, max_len, device)

    # grads_retain = _batch_grad_pairs(
    #     retain_pairs_sampled, encoder, decoder, param_list,
    #     codes_inverse_freq, criterion, output_size, max_len, device)
    
    average_scale = len(unlearn_data_loader) * unlearn_data_loader.batch_size + len(retain_data_loader) * retain_data_loader.batch_size

    grads_forget = _batch_grad(model, unlearn_data_loader, param_list, criterion, average_scale=average_scale)
    grads_retain = _batch_grad(model, retain_data_loader, param_list, criterion, average_scale=average_scale)

    signed_grads = [gr - gf for gr, gf in zip(grads_retain, grads_forget)]

    # scores = torch.tensor([_mean_abs(g).item() for g in signed_grads],
    #                       device=device)
    # k = max(1, int(len(scores) * kookmin_init_rate))
    # thresh = scores.kthvalue(k).values.item()

    # def _reinit_tensor(tensor: torch.Tensor, module: nn.Module, name: str):
    #     with torch.no_grad():
    #         if isinstance(module, nn.Conv2d):
    #             init.kaiming_normal_(tensor)
    #         elif isinstance(module, nn.Linear):
    #             init.kaiming_uniform_(tensor, a=math.sqrt(5))
    #         elif isinstance(module, nn.Embedding):
    #             init.normal_(tensor, 0, 0.02)
    #         elif isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
    #             # weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, ...
    #             if "weight" in name:
    #                 init.xavier_uniform_(tensor)
    #             else:  # bias
    #                 tensor.zero_()
    #         elif isinstance(module, nn.BatchNorm2d):
    #             if "weight" in name:
    #                 tensor.fill_(1.)
    #             else:
    #                 tensor.zero_()
    #         else:
    #             init.normal_(tensor, 0, 0.02)

    # reinit_params, kept_params = [], []

    # print("picking parameters to re-initialize")

    # for (net_name, net) in [("model", model)]:
    #     for n, p in net.named_parameters():
    #         if not p.requires_grad or p.grad is None:
    #             continue

    #         g_idx = param_index[id(p)]       # position in the signed_grad list
    #         if _mean_abs(signed_grads[g_idx]) > thresh:
    #             kept_params.append(p)        # keep, lr will be 0.1·base
    #             continue

    #         module_name = n.split('.')[0]
    #         module = dict(net.named_modules())[module_name]
    #         _reinit_tensor(p, module, n)               
    #         reinit_params.append(p)          # full learning-rate here

    # _reset_adam_state(optimizer, reinit_params)

    # 0) Flatten all signed-grad scores into one vector
    all_scores = torch.cat([g.abs().reshape(-1) for g in signed_grads])
    total_weights = all_scores.numel()
    k = max(1, int(total_weights * kookmin_init_rate))
    thresh = all_scores.kthvalue(k).values.item()

    # 1) Per-weight re-init of the bottom init_rate fraction
    reinit_masks: Dict[nn.Parameter, torch.BoolTensor] = {}

    for (name, p), g in zip(model.named_parameters(), signed_grads):
        # mask of positions we re-initialized
        mask = (g.abs() <= thresh)
        if not mask.any():
            continue

        # actually re-init those positions
        new_p = torch.empty_like(p.data, device=device)
        if p.dim() == 4:
            init.kaiming_normal_(new_p, mode="fan_out", nonlinearity="relu")
        elif p.dim() == 2:
            init.kaiming_uniform_(new_p, a=math.sqrt(5))
        else:
            new_p.normal_(0,0.02)

        p.data = p.data.to(device)
        p.data[mask] = new_p[mask]

        # remember this mask
        reinit_masks[p] = mask

    # reset Adam state on re-init’d params (optional)
    _reset_adam_state(optimizer, list(reinit_masks.keys()))

    # wipe grads so we start clean
    model.zero_grad()

    print("Retain round")

    for epoch in range(retain_epoch_count):
        print(f"Epoch {epoch + 1}/{retain_epoch_count}:")

        print_loss_total = 0

        retain_round_samples = retain_samples_used_for_update
        k_more = max(0, retain_round_samples - len(trainable_cleaned_unlearn_user_ids))
        extra_retain = random.sample(retain_user_ids, k=k_more) if k_more else []
        retain_ids_sampled = trainable_cleaned_unlearn_user_ids + extra_retain

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

        losses, metrics = [], []
        y_true = []
        y_pred = []
        total_loss = 0.0
        tqdm_loader = tqdm.tqdm(retain_data_loader, disable=not LOCAL)
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(tqdm_loader):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            # (B, N)
            output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
            loss = criterion(output, truth_data.float())
            total_loss += loss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            # scale reinitialized params higher, as the lr for all others was set lower
            for p, mask in reinit_masks.items():
                if p.grad is not None:
                    p.grad[mask] *= 10
            optimizer.step()
            y_pred.append(output.detach().cpu())
            y_true.append(truth_data.detach().cpu())
            tqdm_loader.set_description(f'Unlearn epoch: {epoch}, train loss: {total_loss / (step + 1)}')

        # loss_dict[name] = total_loss / (step + 1)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        print(f'metric ...')
        scores = get_metric(y_true=y_true, y_pred=y_pred)
        scores = sorted(scores.items(), key=lambda item: item[0], reverse=False)
        scores = {item[0]: item[1] for item in scores}
        print(json.dumps(scores, indent=4))
        # metric_dict[name] = scores

    sys.stdout.flush()