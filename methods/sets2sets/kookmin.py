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
            opt.state[p]['step'] = 0
            opt.state[p]['exp_avg'].zero_()
            opt.state[p]['exp_avg_sq'].zero_()


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

    scores = torch.tensor([_mean_abs(g).item() for g in signed_grads],
                          device=device)
    k = max(1, int(len(scores) * kookmin_init_rate))
    thresh = scores.kthvalue(k).values.item()

    def _reinit_tensor(tensor: torch.Tensor, module: nn.Module, name: str):
        with torch.no_grad():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(tensor)
            elif isinstance(module, nn.Linear):
                init.kaiming_uniform_(tensor, a=math.sqrt(5))
            elif isinstance(module, nn.Embedding):
                init.normal_(tensor, 0, 0.02)
            elif isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
                # weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, ...
                if "weight" in name:
                    init.xavier_uniform_(tensor)
                else:  # bias
                    tensor.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                if "weight" in name:
                    tensor.fill_(1.)
                else:
                    tensor.zero_()
            else:
                init.normal_(tensor, 0, 0.02)

    reinit_params, kept_params = [], []

    print("picking parameters to re-initialize")

    for (net_name, net) in [("encoder", encoder), ("decoder", decoder)]:
        for n, p in net.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            g_idx = param_index[id(p)]       # position in the signed_grad list
            if _mean_abs(signed_grads[g_idx]) > thresh:
                kept_params.append(p)        # keep, lr will be 0.1·base
                continue

            module_name = n.split('.')[0]
            module = dict(net.named_modules())[module_name]
            _reinit_tensor(p, module, n)               
            reinit_params.append(p)          # full learning-rate here

    _reset_adam_state(encoder_optimizer, reinit_params)
    _reset_adam_state(decoder_optimizer, reinit_params)

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
                        decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size, param_grads_to_scale=kept_params, scale_for_params=0.1)

            print_loss_total += loss


        # print loss and save model
        print_loss_avg = print_loss_total / retain_round_samples
        print_loss_total = 0

        print(f"average loss over {len(unlearning_user_ids)} sample{'s' if len(unlearning_user_ids) != 1 else ''}: {print_loss_avg}")
    sys.stdout.flush()