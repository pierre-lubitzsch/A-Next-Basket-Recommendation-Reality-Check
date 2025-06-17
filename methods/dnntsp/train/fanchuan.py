import torch
import torch.nn as nn
import numpy as np
import sys
from typing import List, Sequence, Tuple, Dict

from tqdm import tqdm
import random


from utils.data_container import get_data_loader, get_data_loader_temporal_split
from utils.util import save_model, convert_to_gpu, convert_all_data_to_gpu
from utils.metric import get_metric


def random_shuffle(arr):
    """
    input:
        arr: any sequence
    
    output:
        random permutation of arr
    """

    arr_permutation = np.random.permutation(len(arr))
    shuffled_arr = []

    for idx in arr_permutation:
        shuffled_arr.append(arr[idx])
    
    return shuffled_arr



def kl_loss_sym(x, y):
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    # kl_loss expects the first parameter to be model outputs as log probabilities and the target to be "normal" probabilities
    return kl_loss(torch.log(x + 1e-20), y)

def unlearn_iterative_uniform_distribution(
    batch,                # one batch from get_data_loader*_…  => (g, nodes_feature, edges_weight, lengths, nodes, truth, freq)
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    optimizer.zero_grad()

    g, nf, ew, lengths, nodes, truth, freq = batch
    g, nf, ew, lengths, nodes, truth, freq = convert_all_data_to_gpu(g, nf, ew, lengths, nodes, truth, freq)

    # forward
    pred = model(g, nf, ew, lengths, nodes, freq)             # (B, items_total)
    # make a uniform target
    uni = torch.full_like(pred, 1.0/pred.size(1))
    loss = kl_loss_sym(pred, uni)

    # backward + step
    loss.backward()
    optimizer.step()

    return loss.item()





# def unlearn_iterative_contrastive(unlearn_variable_batch, retain_variable_batch, encoder, decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer,
#           criterion, output_size, max_length=MAX_LENGTH):
#     encoder_optimizer.zero_grad()

#     unlearn_encoder_batch_outputs = []
#     retain_encoder_batch_outputs = []

#     for unlearn_variable in unlearn_variable_batch:
#         encoder_hidden = encoder.initHidden()
#         input_length = len(unlearn_variable)
#         encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
#         if use_cuda:
#             encoder_outputs = encoder_outputs.cuda()

#         history_record = np.zeros(output_size)
#         for ei in range(input_length - 1):
#             if ei == 0: #because first basket in input variable is [-1]
#                 continue
#             for ele in unlearn_variable[ei]:
#                 history_record[ele] += 1.0 / (input_length - 2)

#         for ei in range(input_length - 1):
#             if ei == 0:
#                 continue
#             encoder_output, encoder_hidden = encoder(unlearn_variable[ei], encoder_hidden)
#             encoder_outputs[ei - 1] = encoder_output[0][0]
        
#         seq_rep = encoder_outputs[: input_length-1, :] # shape: (input_length-1, hidden_size)
#         pooled = seq_rep.mean(dim=0) # shape: (hidden_size,)
#         unlearn_encoder_batch_outputs.append(pooled)
    

#     for retain_variable in retain_variable_batch:
#         encoder_hidden = encoder.initHidden()
#         input_length = len(retain_variable)
#         encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
#         if use_cuda:
#             encoder_outputs = encoder_outputs.cuda()

#         history_record = np.zeros(output_size)
#         for ei in range(input_length - 1):
#             if ei == 0: #because first basket in input variable is [-1]
#                 continue
#             for ele in retain_variable[ei]:
#                 history_record[ele] += 1.0 / (input_length - 2)

#         for ei in range(input_length - 1):
#             if ei == 0:
#                 continue
#             encoder_output, encoder_hidden = encoder(retain_variable[ei], encoder_hidden)
#             encoder_outputs[ei - 1] = encoder_output[0][0]
        
#         seq_rep = encoder_outputs[: input_length-1, :] # shape: (input_length-1, hidden_size)
#         pooled = seq_rep.mean(dim=0) # shape: (hidden_size,)

#         retain_encoder_batch_outputs.append(pooled)

#     unlearn_encoder_batch_outputs = torch.stack(unlearn_encoder_batch_outputs)
#     retain_encoder_batch_outputs = torch.stack(retain_encoder_batch_outputs)

#     if use_cuda:
#         unlearn_encoder_batch_outputs = unlearn_encoder_batch_outputs.cuda()
#         retain_encoder_batch_outputs = retain_encoder_batch_outputs.cuda()

#     t = 1.15 # temperature
#     loss = (-1 * nn.LogSoftmax(dim=-1)(unlearn_encoder_batch_outputs @ retain_encoder_batch_outputs.T / t)).mean()
#     loss.backward()

#     encoder_optimizer.step()

#     return loss.item()

# def unlearn_iterative_contrastive(
#     unlearn_batch: Sequence,   # list of graph‐tuples
#     retain_batch: Sequence,
#     model: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     temperature: float = 1.15,
# ):
#     model.train()
#     optimizer.zero_grad()

#     # pass each batch through the model to get pooled embeddings
#     def encode_pool(batch):
#         embs = []
#         for tup in batch:
#             g, nf, ew, lengths, nodes, truth, freq = convert_all_data_to_gpu(*tup)
#             out = model(g, nf, ew, lengths, nodes, freq)          # (1, items_total)
#             # mean‐pool across items axis
#             embs.append(out.mean(dim=1))
#         return torch.cat(embs, dim=0)                              # (N, hidden_dim)

#     U = encode_pool(unlearn_batch)                               # (u, D)
#     R = encode_pool(retain_batch)                                 # (r, D)

#     sims = (U @ R.T) / temperature
#     loss = - nn.LogSoftmax(dim=1)(sims).mean()                    # cross‐contrastive
#     loss.backward()
#     optimizer.step()
#     return loss.item()



def unlearn_iterative_contrastive(
    unb: Tuple,
    retb: Tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    temperature: float = 1.15,
):
    model.train()
    optimizer.zero_grad()

    def encode(batch):
        g,nf,ew,lengths,nodes,truth,freq = convert_all_data_to_gpu(*batch)
        _, embedding = model(g, nf, ew, lengths, nodes, freq, return_embedding=True)  # [B, D]
        return embedding                      # [B]

    U = encode(unb)  # shape (1,)
    R = encode(retb) # shape (16,)

    # if D>1 use U @ R.T, but for scalars it broadcast‐multiplies
    sims = (U.unsqueeze(1) * R.unsqueeze(0)) / temperature  
    loss = -nn.LogSoftmax(dim=1)(sims).mean()
    loss.backward()
    optimizer.step()
    return loss.item()




# def unlearn_neurips_competition_iterative_contrastive(unlearning_user_ids, retain_user_ids, clean_data_history_and_future, data_history, data_future, encoder, decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size, start, n_iters, constrastive_retain_batchsize=16, LOCAL=False, temporal_split=True, print_loss_total=0, total_iter=0, best_recall=0, unlearn_iters_contrastive = 8):
#     total_steps_expected = len(unlearning_user_ids) * (1 + unlearn_iters_contrastive + 10)

#     print("First stage: learn uniform pseudolabel")
#     # get a suffle list
#     user_permutation = np.random.permutation(len(unlearning_user_ids))
#     shuffled_users = []
#     for idx in user_permutation:
#         shuffled_users.append(unlearning_user_ids[idx])

#     for iter in tqdm(range(0, len(unlearning_user_ids)), disable=not LOCAL):
#         # get training data and label.
#         # get a train batch
#         if temporal_split:
#             # skip user if their basket count is too small for having at least 2 training, 1 valid, and 1 test basket
#             # substract 2 for the real lengths because the basket lists are padded with [-1] at the start and end
#             assert len(data_history[shuffled_users[iter]]) - 2 + len(data_future[shuffled_users[iter]]) - 2 >= 4, f"the basket count for user {shuffled_users[iter]} is smaller than 4, but he was not filtered out"
#             # data_history[training_keys[iter]][-3] is the training label
#             # data_history[training_keys[iter]][-2] is the valid label
#             # data_future[training_keys[iter]][1] is the test label
#             prev_target_variable = [[-1], data_history[shuffled_users[iter]][-3], [-1]]
#             prev_input_variable = data_history[shuffled_users[iter]][:-3] + [[-1]]
#             # potential to have "retain sample" with the unlearning items removed from the basket sequence
#             # for now just do the method on the data as is
#         else:
#             input_variable = data_history[shuffled_users[iter]]
#             target_variable = data_future[shuffled_users[iter]]

#         loss = unlearn_iterative_uniform_distribution(prev_input_variable, encoder,
#                     decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size)

#         print_loss_total += loss
#         total_iter += 1

#     print_loss_avg = print_loss_total / len(unlearning_user_ids)
#     print_loss_total = 0
#     # print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (n_iters * len(unlearning_user_ids))), total_iter,
#     #                             total_iter / (n_iters * len(unlearning_user_ids)) * 100, print_loss_avg))
#     # print_progress(start, total_iter, total_steps_expected, print_loss_avg)
#     print(f"average loss over {len(unlearning_user_ids)} sample{"s" if len(unlearning_user_ids) != 1 else ""}: {print_loss_avg}")
#     sys.stdout.flush()

#     print("Second stage: learn uniform pseudolabel")
#     # Second stage: 
#     for j in range(unlearn_iters_contrastive):
#         print_loss_total = 0
#         # Contrastive round

#         shuffled_unlearn_users = random_shuffle(unlearning_user_ids)

#         for iter in tqdm(range(0, len(unlearning_user_ids)), disable=not LOCAL):
#             retain_batch_user_ids = random.sample(retain_user_ids, k=constrastive_retain_batchsize)
#             retain_input_batch = [data_history[user][:-3] + [[-1]] for user in retain_batch_user_ids]

#             if temporal_split:
#                 # skip user if their basket count is too small for having at least 2 training, 1 valid, and 1 test basket
#                 # substract 2 for the real lengths because the basket lists are padded with [-1] at the start and end
#                 assert len(data_history[shuffled_unlearn_users[iter]]) - 2 + len(data_future[shuffled_unlearn_users[iter]]) - 2 >= 4, f"the basket count for user {shuffled_unlearn_users[iter]} is smaller than 4, but he was not filtered out"
#                 # data_history[training_keys[iter]][-3] is the training label
#                 # data_history[training_keys[iter]][-2] is the valid label
#                 # data_future[training_keys[iter]][1] is the test label
#                 prev_target_variable = [[-1], data_history[shuffled_unlearn_users[iter]][-3], [-1]]
#                 prev_input_variable = data_history[shuffled_unlearn_users[iter]][:-3] + [[-1]]
#                 # potential to have "retain sample" with the unlearning items removed from the basket sequence
#                 # for now just do the method on the data as is
#             else:
#                 # not fully implemented
#                 input_variable = data_history[retain_batch_user_ids[0]]
#                 target_variable = data_future[retain_batch_user_ids[0]]

#             unlearn_input_batch = [prev_input_variable]
#             loss = unlearn_iterative_contrastive(unlearn_input_batch, retain_input_batch, encoder,
#                         decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size)

#             print_loss_total += loss
#             total_iter += 1

#         # key_idx = np.random.permutation(len(training_key_set))
#         # for idx in tqdm(key_idx):
#         #     input_variable, target_variable = user_to_training_data[training_key_set[idx]]

#         #     loss = train(input_variable, target_variable, encoder,
#         #                  decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size)

#         #     print_loss_total += loss
#         #     total_iter += 1

#         # print loss and save model
#         print_loss_avg = print_loss_total / len(unlearning_user_ids)
#         print_loss_total = 0
#         # print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (n_iters * len(unlearning_user_ids))), total_iter,
#         #                             total_iter / (n_iters * len(unlearning_user_ids)) * 100, print_loss_avg))
#         # print_progress(start, total_iter, total_steps_expected, print_loss_avg)
#         print(f"average loss over {len(unlearning_user_ids)} sample{"s" if len(unlearning_user_ids) != 1 else ""}: {print_loss_avg}")
#         sys.stdout.flush()


#         # Retain round

#         print_loss_total = 0
#         # use all user ids for retain round but filter out unlearned items from respective users basket histories and labels
#         retain_round_user_ids = list(clean_data_history_and_future.keys())
#         retain_round_samples = 10 * len(unlearning_user_ids)
#         clean_sample_users = [u for u in unlearning_user_ids if u in clean_data_history_and_future.keys()]
#         shuffled_users = clean_sample_users + random.sample(retain_round_user_ids, k=retain_round_samples - len(clean_sample_users))

#         for iter in tqdm(range(retain_round_samples), disable=not LOCAL):
#             user = shuffled_users[iter]
#             basket_history = clean_data_history_and_future[user]
#             # get training data and label.
#             # get a train batch
#             if temporal_split:
#                 # basket_history is in the format [[-1], b_1, ..., b_n, [-1]] with the complete basket history b_1, ..., b_n
#                 # assert len(basket_history) >= 6, f"{basket_history} is too short"
#                 target_variable = [[-1], basket_history[-4], [-1]]
#                 input_variable = basket_history[:-4] + [[-1]]
#                 # potential to have "retain sample" with the unlearning items removed from the basket sequence
#                 # for now just do the method on the data as is
#             else:
#                 input_variable = data_history[shuffled_users[iter]]
#                 target_variable = data_future[shuffled_users[iter]]

#             loss = train(input_variable, target_variable, encoder,
#                         decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size)

#             print_loss_total += loss
#             total_iter += 1

#         # key_idx = np.random.permutation(len(training_key_set))
#         # for idx in tqdm(key_idx):
#         #     input_variable, target_variable = user_to_training_data[training_key_set[idx]]

#         #     loss = train(input_variable, target_variable, encoder,
#         #                  decoder, codes_inverse_freq, encoder_optimizer, decoder_optimizer, criterion, output_size)

#         #     print_loss_total += loss
#         #     total_iter += 1

#         # print loss and save model
#         print_loss_avg = print_loss_total / retain_round_samples
#         print_loss_total = 0
#         # print('%s (%d %d%%) %.6f' % (timeSince(start, total_iter / (n_iters * len(unlearning_user_ids))), total_iter,
#         #                             total_iter / (n_iters * len(unlearning_user_ids)) * 100, print_loss_avg))
#         # print_progress(start, total_iter, total_steps_expected, print_loss_avg)
#         print(f"average loss over {len(unlearning_user_ids)} sample{"s" if len(unlearning_user_ids) != 1 else ""}: {print_loss_avg}")
#         sys.stdout.flush()

def batch_to_samples(batch):
    # batch: (g, nf, ew, lengths, nodes, truth, freq)
    g, nf, ew, lengths, nodes, truth, freq = batch
    B = truth.size(0)
    return [
        (g[i], nf[i], ew[i], lengths[i], nodes[i], truth[i], freq[i])
        for i in range(B)
    ]


def unlearn_neurips_iterative_contrastive(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    uniform_iters: int = 1,
    contrastive_iters: int = 8,
    retain_iters: int = 10,
    retain_batch_size: int = 16,
    device: str = "cuda",
    LOCAL: bool = False,
    history_path=None,
    future_path=None,
    unlearning_user_ids=None,
    retain_user_ids=None,
    user_to_unlearning_items=None,
    trainable_cleaned_unlearn_user_ids=None,
):
    unlearn_loader = get_data_loader_temporal_split(history_path=history_path,
                                    future_path=future_path,
                                    data_type='train',
                                    batch_size=1,
                                    item_embedding_matrix=model.item_embedding,
                                    retrain_flag=False,
                                    unlearn_set_flag=True,
                                    users_in_unlearning_set=unlearning_user_ids,
                                    user_to_unlearning_items=user_to_unlearning_items,)

    print("Stage 1: uniform pseudolabel")
    # uniform distribution
    for _ in tqdm(range(uniform_iters), disable=not LOCAL):
        for batch in unlearn_loader:
            unlearn_iterative_uniform_distribution(
                batch, model, optimizer
            )

    print("Stage 2: contrastive unlearning & Fine-tuning")
    for j in tqdm(range(contrastive_iters), disable=not LOCAL):
        contrastive_retain_batch_size = 16
        retain_ids_sampled = random.sample(retain_user_ids, k=contrastive_retain_batch_size)
        retain_loader = get_data_loader_temporal_split(history_path=history_path,
                                                future_path=future_path,
                                                data_type='train',
                                                batch_size=16,
                                                item_embedding_matrix=model.item_embedding,
                                                retrain_flag=True,
                                                unlearn_set_flag=False,
                                                users_in_unlearning_set=unlearning_user_ids,
                                                user_to_unlearning_items=user_to_unlearning_items,
                                                user_subset=retain_ids_sampled,)
        for unb, retb in zip(unlearn_loader, retain_loader):
            unlearn_iterative_contrastive(unb, retb, model, optimizer)
        
        # retain round

        k_missing = 10 * len(unlearning_user_ids) - len(trainable_cleaned_unlearn_user_ids)
        retain_ids_sampled = random.sample(retain_user_ids, k=k_missing)
        retain_ids_to_take = trainable_cleaned_unlearn_user_ids + retain_ids_sampled
        retain_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=retain_batch_size,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=True,
                                            unlearn_set_flag=False,
                                            users_in_unlearning_set=unlearning_user_ids,
                                            user_to_unlearning_items=user_to_unlearning_items,
                                            user_subset=retain_ids_to_take,
                                            shuffle=True)
        
        model.train()
        for batch in retain_loader:
            g, nf, ew, lengths, nodes, truth, freq = batch
            g, nf, ew, lengths, nodes, truth, freq = convert_all_data_to_gpu(g, nf, ew, lengths, nodes, truth, freq)

            optimizer.zero_grad()
            pred = model(g, nf, ew, lengths, nodes, freq)
            loss = nn.MultiLabelSoftMarginLoss()(pred, truth.float())
            loss.backward()
            optimizer.step()
