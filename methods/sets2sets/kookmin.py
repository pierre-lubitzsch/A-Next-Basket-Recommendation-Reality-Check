import math, random
from typing import List, Sequence, Tuple, Dict

import torch, torch.nn as nn, torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import tqdm
import sys
import tqdm

from sets2sets_new import train


def _loss_forward(
    encoder, decoder, input_var, target_var,
    codes_inverse_freq, criterion, output_size, max_len
):
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

def _batch_grad_pairs(pairs, encoder, decoder, param_list,
                      codes_inverse_freq, criterion,
                      output_size, max_len, device) -> List[torch.Tensor]:
    """Average gradient over a list of (inp,tgt) pairs."""
    acc = [torch.zeros_like(p, device=device) for p in param_list]
    for inp, tgt in pairs:
        loss = _loss_forward(encoder, decoder, inp, tgt,
                             codes_inverse_freq, criterion,
                             output_size, max_len)
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=False,
            allow_unused=True
        )
        # fill in zeros for any None
        grads = [
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(param_list, grads)
        ]
        for a, g in zip(acc, grads):
            a += g.detach()
    for a in acc:                        # mean
        a /= max(1, len(pairs))
    return acc


def _mean_abs(t: torch.Tensor) -> torch.Tensor:
    return t.abs().mean()


def _reset_adam_state(opt: torch.optim.Adam, params: List[torch.Tensor]):
    for p in params:
        if p in opt.state:
            opt.state[p]['step'] = torch.zeros(1, dtype=torch.float32, device=p.device)
            opt.state[p]['exp_avg'].zero_()
            opt.state[p]['exp_avg_sq'].zero_()


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def unlearn_by_reinit_and_finetune(
    *,
    unlearning_user_ids: List[str],
    retain_user_ids: List[str],
    cur_clean_data_history_and_future: Dict[str, list],
    history_data: Dict[str, list],
    future_data: Dict[str, list],
    encoder: nn.Module,
    decoder: nn.Module,
    codes_inverse_freq: np.ndarray,
    criterion: nn.Module,
    output_size: int,
    LOCAL: bool,
    temporal_split: bool = True,
    kookmin_init_rate: float = 0.01,     # 1 % of lowest-|grad| params
    device: str = "cuda",
    retain_pairs=None,                   # list of (inp,tgt) tuples
    neg_grad_retain_sample_size: int = 128,
    max_len: int = 100,
    retain_epoch_count=5,
    decoder_optimizer=None,
    encoder_optimizer=None,
    param_list=None,
    param_index=None,
    retain_samples_used_for_update=32,
):
    """
    Re-implements Kookmin’s low-|grad| re-init but with *encoder/decoder*
    instead of a monolithic model.
    """

    encoder.to(device).train()
    decoder.to(device).train()

    move_optimizer_state(encoder_optimizer, device)
    move_optimizer_state(decoder_optimizer, device)

    def make_pair(u, sensitive_included):
        if temporal_split:
            if sensitive_included:  # forget sample
                tgt = [[-1], history_data[u][-3], [-1]]
                inp = history_data[u][:-3] + [[-1]]
            else:                  # “clean” version of the same user
                tgt = [[-1], cur_clean_data_history_and_future[u][-4], [-1]]
                inp = cur_clean_data_history_and_future[u][:-4] + [[-1]]
        else:
            inp, tgt = history_data[u], future_data[u]
        return inp, tgt

    forget_pairs = [make_pair(u, True) for u in unlearning_user_ids]

    clean_unlearn_pairs = [
        make_pair(u, False)
        for u in unlearning_user_ids
        if u in cur_clean_data_history_and_future
    ]

    # sample additional retain pairs so that we have
    # `neg_grad_retain_sample_size` in total
    k_more = max(0, neg_grad_retain_sample_size - len(clean_unlearn_pairs))
    extra_retain = random.sample(retain_pairs, k=k_more) if k_more else []
    retain_pairs_sampled = clean_unlearn_pairs + extra_retain


    grads_forget = _batch_grad_pairs(
        forget_pairs, encoder, decoder, param_list,
        codes_inverse_freq, criterion, output_size, max_len, device)

    grads_retain = _batch_grad_pairs(
        retain_pairs_sampled, encoder, decoder, param_list,
        codes_inverse_freq, criterion, output_size, max_len, device)

    signed_grads = [gr - gf for gr, gf in zip(grads_retain, grads_forget)]

    all_scores = torch.cat([g.abs().reshape(-1) for g in signed_grads])
    total = all_scores.numel()
    k = max(1, int(total * kookmin_init_rate))
    thresh = all_scores.kthvalue(k).values.item()

    # 3) re-init only the low-gradient entries *and* remember a mask per param
    reinit_masks: Dict[torch.Tensor, torch.BoolTensor] = {}
    for p, g in zip(param_list, signed_grads):
        mask = g.abs() <= thresh
        if not mask.any():
            continue

        # make a freshly initialized tensor of the same shape
        new_p = torch.empty_like(p.data, device=device)
        if p.dim() == 4:            # e.g. Conv2d weight
            nn.init.kaiming_normal_(new_p, mode="fan_out", nonlinearity="relu")
        elif p.dim() == 2:          # e.g. Linear weight
            nn.init.kaiming_uniform_(new_p, a=math.sqrt(5))
        else:                       # embeddings, biases, …
            new_p.normal_(0, 0.02)

        # overwrite only the “low-grad” slots
        p.data = p.data.to(device)
        p.data[mask] = new_p[mask]

        # store the mask to use later
        reinit_masks[p] = mask

    # 4) reset Adam state on exactly those params
    _reset_adam_state(encoder_optimizer, list(reinit_masks.keys()))
    _reset_adam_state(decoder_optimizer, list(reinit_masks.keys()))

    # wipe grads so we start clean
    encoder.zero_grad()
    decoder.zero_grad()

    print("Retain round")

    for epoch in range(retain_epoch_count):
        print(f"Epoch {epoch + 1}/{retain_epoch_count}:")

        print_loss_total = 0

        retain_round_samples = retain_samples_used_for_update
        k_more = max(0, retain_round_samples - len(clean_unlearn_pairs))
        extra_retain = random.sample(retain_pairs, k=k_more) if k_more else []
        retain_pairs_sampled = clean_unlearn_pairs + extra_retain

        for input_variable, target_variable in tqdm.tqdm(retain_pairs_sampled, disable=not LOCAL):
            loss = train(input_variable, target_variable, encoder,
                        decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size, reinit_masks=reinit_masks, scale_for_reinit_params=10)

            print_loss_total += loss


        # print loss and save model
        print_loss_avg = print_loss_total / retain_round_samples
        print_loss_total = 0

        print(f"average loss over {len(unlearning_user_ids)} sample{'s' if len(unlearning_user_ids) != 1 else ''}: {print_loss_avg}")
    sys.stdout.flush()