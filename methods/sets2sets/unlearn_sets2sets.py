from sets2sets_new import *
import sys
import pickle
import argparse
import torch
import scif
import kookmin
import fanchuan


learning_rate = 0.001



def parse_args():
    parser = argparse.ArgumentParser(description="Process dataset arguments for training/unlearning.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="tafeng",
        help="Dataset name"
    )
    parser.add_argument(
        "--ind",
        type=int,
        default=0,
        help="Index for the keyset and model version"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k parameter"
    )
    parser.add_argument(
        "--training",
        type=int,
        required=True,
        help="1 for retraining, 2 for unlearning"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--temporal_split",
        action="store_true",
        help="Temporal split flag. If not set do user split"
    )
    parser.add_argument(
        "--LOCAL",
        action="store_true",
        help="Local flag. If not set assume we run on cluster"
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=0.0001,
        help="Fraction of data to unlearn"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="popular",
        help="Unlearning method to use: random, popular, unpopular, or sensitive"
    )
    parser.add_argument(
        "--popular_percentage",
        type=float,
        default=0.1,
        help="Fraction of most/least popular items to consider"
    )
    parser.add_argument(
        "--unlearning_algorithm",
        type=str,
        default="scif",
        choices=[
            "neurips_competition_iterative_contrastive",
            "scif",
            "kookmin",
        ],
        help="What unlearning algorithm is used."
    )
    parser.add_argument(
        "--sensitive_category",
        type=str,
        default=None,
        choices=["baby", "meat", "alcohol"],
        help="When choosing sensitive items to unlearn, choose which category"
    )
    parser.add_argument(
        "--retrain_checkpoint_idx_to_match",
        type=int,
        default=None,
        help="which unlearning checkpoint should be taken as example to create the unlearning set (only take a subset of the unlearning set given)"
    )
    parser.add_argument(
        "--lissa_train_pair_count_scif",
        type=int,
        default=1024,
        help="how many samples are used for lissa hessian estimation"
    )
    parser.add_argument(
        "--retain_samples_used_for_update",
        type=int,
        default=128,
        help="how many samples are used in the HVP Hv inside v (v is the avg of the gradients of the unlearn sample, the cleaned one and some retain samples)"
    )
    parser.add_argument(
        "--kookmin_init_rate",
        type=float,
        default=0.01,
        help="percentage of parameters getting reset in the kookmin unlearning algorithm",
    )

    return parser.parse_args()




def make_pair(u, sensitive_included, temporal_split, history_data, future_data, cur_clean_data_history_and_future):
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

def print_progress(start, finished, total, loss_avg):
    """Pretty progress + ETA."""
    elapsed = time.time() - start
    pct     = finished / max(total, 1)
    eta     = elapsed * (1 - pct) / max(pct, 1e-8)

    def fmt(t): m, s = divmod(int(t), 60); return f"{m}m {s}s"
    print(f"{fmt(elapsed)} (- {fmt(eta)}) "
          f"({finished}/{total}  {pct*100:5.1f}%) "
          f"{loss_avg:.6f}")


# --- collect the model files we just trained ---------------------------
def _find_trained_models(args, model_version, seed):
    """
    Returns two equal-length lists with encoder/decoder paths
    created in *this* run, in ascending epoch order.
    """
    enc_paths, dec_paths = [], []

    if args.training == 1:                                    # retraining
        retrain_tag = (
            f"_sensitive_category_{args.sensitive_category}"
            f"_unlearning_fraction_{args.unlearning_fraction}"
            f"_retrain_checkpoint_idx_to_match_{args.retrain_checkpoint_idx_to_match}"
        )
        enc = f"./models/encoder_{model_version}_model_best_seed_{seed}{retrain_tag}.pt"
        dec = f"./models/decoder_{model_version}_model_best_seed_{seed}{retrain_tag}.pt"
        if os.path.exists(enc) and os.path.exists(dec):
            enc_paths.append(enc)
            dec_paths.append(dec)

    elif args.training == 2:                                  # unlearning
        unlearn_tag = (
            f"_sensitive_category_{args.sensitive_category}"
            f"_unlearning_fraction_{args.unlearning_fraction}"
            f"_unlearning_algorithm_{args.unlearning_algorithm}"
        )
        pattern = f"unlearn_encoder_{model_version}_model_best_unlearn_epoch"
        print(pattern)
        for fname in sorted(os.listdir("./models")):
            if fname.startswith(pattern) and fname.endswith(f"_seed_{seed}{unlearn_tag}.pt"):
                epoch = fname.split("unlearn_epoch")[1].split("_")[0]
                enc = f"./models/{fname}"
                dec = enc.replace("unlearn_encoder_", "unlearn_decoder_")
                if os.path.exists(dec):
                    enc_paths.append(enc)
                    dec_paths.append(dec)

    return enc_paths, dec_paths
# ---------------------------------------------------------------------------


# --- generic evaluation routine --------------------------------------
def _evaluate_and_print(paths, history_data, future_data, input_size,
                        val_users, test_users, next_k_step,
                        topk_list, temporal_split):
    """
    paths: list[(encoder_path, decoder_path)]
    prints results for every (enc, dec) pair found in *paths*.
    """
    for k in topk_list:
        print("=" * 80)
        print(f" Top-{k} evaluation")
        print("=" * 80)
        for idx, (enc_p, dec_p) in enumerate(paths):
            print(f"[{idx}] Model {os.path.basename(enc_p)}")

            encoder = torch.load(enc_p, map_location=torch.device("cuda" if use_cuda else "cpu"), weights_only=False)
            decoder = torch.load(dec_p, map_location=torch.device("cuda" if use_cuda else "cpu"), weights_only=False)

            with torch.no_grad():
                val_metrics  = evaluate(history_data, future_data, encoder, decoder,
                                        input_size, val_users,  next_k_step, k,
                                        test_flag=True, temporal_split=temporal_split)
                test_metrics = evaluate(history_data, future_data, encoder, decoder,
                                        input_size, test_users, next_k_step, k,
                                        test_flag=True, temporal_split=temporal_split)

            vr, vn, vh  = val_metrics
            tr, tn, th  = test_metrics
            print(f"  • VAL   – Recall:{vr:.4f}  NDCG:{vn:.4f}  HR:{vh:.4f}")
            print(f"  • TEST  – Recall:{tr:.4f}  NDCG:{tn:.4f}  HR:{th:.4f}")
        print()  # blank line after each k
# ---------------------------------------------------------------------------






def unlearnIters(data_history, data_future, output_size, encoder, decoder, model_name, training_key_set, val_keyset, retain_key_set, codes_inverse_freq, next_k_step,
               n_iters, top_k, seed, temporal_split, LOCAL, user_to_unlearning_items, unlearning_algorithm, constrastive_retain_batchsize=16, args=None, learning_rate=0.001):
    start = time.time()
    start_perf = time.perf_counter()
    print_loss_total = 0  # Reset every print_every
    # elem_wise_connection.initWeight()

    # scale down learning rate for all parameters. when we reinitialize some parameters it gets scaled up for them
    if unlearning_algorithm == "kookmin":
        learning_rate *= 0.1
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                         weight_decay=0)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-11,
                                         weight_decay=0)


    total_iter = 0
    criterion = custom_MultiLabelLoss_torch()
    best_recall = 0.0

    # if temporal_split:
    #     user_to_training_data = {user: (data_history[user][:-3] + [[-1]], [[-1], data_history[user][-3], [-1]]) for user in training_key_set}
    # else:
    #     user_to_training_data = {user: (data_history[user], data_future[user]) for user in training_key_set}

    unlearning_user_ids = training_key_set
    retain_user_ids = retain_key_set

    # create versions of baskets without unlearning items
    clean_data_history_and_future = dict()
    for user in unlearning_user_ids:
        basket_history = data_history[user][1:-1] + [data_future[user][1]]
        clean_history = []

        for basket in basket_history:
            new_basket = list(filter(lambda item: item not in user_to_unlearning_items[user], basket))
            if len(new_basket) > 0:
                clean_history.append(new_basket)
        
        if len(clean_history) < 4:
            if LOCAL:
                print(f"only {len(clean_history)} baskets left in the history of user {user} after deleting the unlearning items -> don't use in retain round")
            continue

        clean_data_history_and_future[user] = [[-1]] + clean_history + [[-1]]
    
    for user in retain_user_ids:
        clean_data_history_and_future[user] = data_history[user][:-1] + [data_future[user][1], [-1]]

    # unlearn sequentually per user and save every quarter of len(unlearning_user_ids)
    n = len(unlearning_user_ids)
    checkpoint_every = math.ceil(n / 4)
    checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]
    if len(checkpoint_idxs) == 5:
        checkpoint_idxs = checkpoint_idxs[:4] + [checkpoint_idxs[-1]]

    cur_clean_data_history_and_future = {u: baskets for u, baskets in clean_data_history_and_future.items() if u in retain_user_ids}
    retain_pairs = [make_pair(u, True, temporal_split, data_history, data_future, cur_clean_data_history_and_future) for u in retain_user_ids]

    # needed for kookmin
    param_list = [p for p in encoder.parameters()
            if p.requires_grad] + \
            [p for p in decoder.parameters()
            if p.requires_grad]
    param_index = {id(p): i for i,p in enumerate(param_list)}

    for i, user in enumerate(sorted(unlearning_user_ids)):
        print(f"\nunlearning items for user {i + 1}/{len(unlearning_user_ids)} with id: {user}\n")
        cur_unlearning_user_ids = [user]
        if user in clean_data_history_and_future.keys():
            cur_clean_data_history_and_future[user] = clean_data_history_and_future[user]

        if unlearning_algorithm == "neurips_competition_iterative_contrastive":
            fanchuan.unlearn_neurips_competition_iterative_contrastive(
                cur_unlearning_user_ids,
                retain_user_ids,
                cur_clean_data_history_and_future,
                data_history,
                data_future,
                encoder,
                decoder,
                codes_inverse_freq,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
                output_size,
                start,
                n_iters,
                constrastive_retain_batchsize=constrastive_retain_batchsize,
                LOCAL=LOCAL,
                temporal_split=temporal_split,
                print_loss_total=print_loss_total,
                total_iter=total_iter,
                best_recall=best_recall
            )
        elif unlearning_algorithm == "scif":
            scif.scif_unlearn(
                unlearning_user_ids=cur_unlearning_user_ids,
                retain_user_ids=retain_user_ids,
                cur_clean_data_history_and_future=cur_clean_data_history_and_future,
                history_data=data_history,
                future_data=data_future,
                encoder=encoder,
                decoder=decoder,
                codes_inverse_freq=codes_inverse_freq,
                criterion=criterion,
                output_size=output_size,
                LOCAL=LOCAL,
                temporal_split=temporal_split,
                retain_pairs=retain_pairs,
                train_pair_count=args.lissa_train_pair_count_scif,
                retain_samples_used_for_update=args.retain_samples_used_for_update,
            )
        elif unlearning_algorithm == "kookmin":
            kookmin.unlearn_by_reinit_and_finetune(
                unlearning_user_ids=cur_unlearning_user_ids,
                retain_user_ids=retain_user_ids,
                cur_clean_data_history_and_future=cur_clean_data_history_and_future,
                history_data=data_history,
                future_data=data_future,
                encoder=encoder,
                decoder=decoder,
                codes_inverse_freq=codes_inverse_freq,
                criterion=criterion,
                output_size=output_size,
                LOCAL=LOCAL,
                temporal_split=temporal_split,
                retain_pairs=retain_pairs,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                kookmin_init_rate=args.kookmin_init_rate,
                device=torch.device("cuda" if use_cuda else "cpu"),
                param_list=param_list,
                param_index=param_index,
                retain_samples_used_for_update=args.retain_samples_used_for_update,
            )
        else:
            print(f"Invalid unlearning algorithm: {unlearning_algorithm}.")
            exit(1)

        if user in clean_data_history_and_future.keys():
            del cur_clean_data_history_and_future[user]

        if i in checkpoint_idxs:
            unlearn_str = (
                f"_sensitive_category_{args.sensitive_category}"
                f"_unlearning_fraction_{args.unlearning_fraction}"
                f"_unlearning_algorithm_{args.unlearning_algorithm}"
            )
            torch.save(
                encoder,
                f"./models/unlearn_encoder_{model_name}_model_best_unlearn_epoch{i}_seed_{seed}{unlearn_str}.pt"
            )
            torch.save(
                decoder,
                f"./models/unlearn_decoder_{model_name}_model_best_unlearn_epoch{i}_seed_{seed}{unlearn_str}.pt"
            )

        print("elapsed {:.2f}s".format(time.perf_counter() - start_perf))
        sys.stdout.flush()
    




def unlearn_main():
    args = parse_args()
    directory = './amodels/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    dataset = args.dataset
    ind = args.ind
    history_file = '../../jsondata/'+dataset+'_history.json'
    future_file = '../../jsondata/'+dataset+'_future.json'
    keyset_file = '../../keyset/'+dataset+'_keyset_'+str(ind)+'.json'
    model_version = dataset+str(ind)
    topk = args.topk
    training = args.training
    seed = args.seed
    temporal_split = args.temporal_split
    LOCAL = args.LOCAL
    unlearning_fraction = args.unlearning_fraction
    method = args.method
    popular_percentage = args.popular_percentage
    use_cuda = not LOCAL
    unlearning_algorithm = args.unlearning_algorithm
    sensitive_category = args.sensitive_category

    percentage_str = f"_popular_percentage_{args.popular_percentage}" if args.method in ["popular", "unpopular"] else ""
    fraction_str = f"_unlearning_fraction_{args.unlearning_fraction}" if args.method in ["popular", "unpopular", "random", "sensitive"] else ""
    unlearning_data_file = f'../../unlearning_data/dataset_{dataset}_seed_{args.seed}_method_{method}{fraction_str}{percentage_str}.pkl'

    set_seed(seed)

    next_k_step = 1
    with open(history_file, 'r') as f:
        history_data = json.load(f)
    with open(future_file, 'r') as f:
        future_data = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
    with open(unlearning_data_file, "rb") as f:
        user_to_unlearning_items = pickle.load(f)
        if method == "sensitive":
            user_to_unlearning_items = user_to_unlearning_items[sensitive_category]

    if temporal_split:
        # input_size, a.k.a. the number of items, is the same for all, and as it is saved only in the keyset file as of now we get it from there
        input_size = keyset['item_num']
        # skip user if their basket count is too small for having at least 2 training, 1 valid, and 1 test basket
        # substract 2 for the real lengths because the basket lists are padded with [-1] at the start and end
        user_list = list(filter(lambda x: len(history_data[x]) - 2 + len(future_data[x]) - 2 >= 4, list(future_data.keys())))
        training_key_set = list(user_to_unlearning_items.keys())
        assert set(training_key_set).issubset(set(user_list)), f"unlearning set contains users which are not in the filtered user list. users not in the filtered list but in the unlearning list:\n\n{set(training_key_set) - set(user_list)}"
        val_key_set = user_list.copy()
        test_key_set = user_list.copy()
        retain_key_set = list(set(user_list) - set(training_key_set))
    else:
        input_size = keyset['item_num']
        training_key_set = keyset['train']
        val_key_set = keyset['val']
        test_key_set = keyset['test']

    # weights is inverse personal top frequency. normalized by max freq.
    weights = np.zeros(input_size)
    codes_freq = get_codes_frequency_no_vector(history_data, input_size, future_data.keys())
    max_freq = max(codes_freq)
    for idx in range(len(codes_freq)):
        if codes_freq[idx] > 0:
            weights[idx] = max_freq / codes_freq[idx]
        else:
            weights[idx] = 0

    # Sets2sets model
    encoder = EncoderRNN_new(input_size, hidden_size, num_layers)
    attn_decoder = AttnDecoderRNN_new(hidden_size, input_size, num_layers, dropout_p=0.1)
    if use_cuda:
        encoder = encoder.cuda()
        attn_decoder = attn_decoder.cuda()

    if training == 1: # retrain
        retrain_str = (
            f"_sensitive_category_{args.sensitive_category}"
            f"_unlearning_fraction_{args.unlearning_fraction}"
            f"_retrain_checkpoint_idx_to_match_{args.retrain_checkpoint_idx_to_match}"
        )
        # remove current forget set from the training data. need parameter to tell how much of the forget set is taken to get the wanted retrained models at certain subsets of the unlearning set
        n = len(training_key_set)
        checkpoint_every = math.ceil(n / 4)
        checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]
        print(checkpoint_idxs)
        if len(checkpoint_idxs) == 5:
            checkpoint_idxs = checkpoint_idxs[:4] + [checkpoint_idxs[-1]]

        unlearning_set_take_first_x = checkpoint_idxs[args.retrain_checkpoint_idx_to_match]
        # remove sensitive items from users in retraining
        users_in_unlearning_set = sorted(training_key_set)[:unlearning_set_take_first_x + 1]
        filtered_users_in_unlearning_set_no_adaptation = []
        for user in users_in_unlearning_set:
            unpadded_baskets = history_data[user][1:-1] + [future_data[user][1]]
            clean_unpadded_baskets = [[item for item in basket if item not in user_to_unlearning_items[user]] for basket in unpadded_baskets]
            clean_unpadded_baskets = list(filter(lambda x: len(x) > 0, clean_unpadded_baskets))
            if len(clean_unpadded_baskets) < 4:
                filtered_users_in_unlearning_set_no_adaptation.append(user)
                continue
            history_data[user] = [[-1]] + clean_unpadded_baskets[:-1] + [[-1]]
            future_data[user] = [[-1], clean_unpadded_baskets[-1], [-1]]
            assert len(history_data[user]) - 2 + len(future_data[user]) - 2 >= 4, f"{history_data}, {future_data} have not enough baskets (< 4 non-dummy)"

        # don't train on users where if we take away
        training_key_set = sorted(set(user_list) - set(filtered_users_in_unlearning_set_no_adaptation))
        trainIters(history_data, future_data, input_size, encoder, attn_decoder, model_version,
                   training_key_set, val_key_set, weights,
                   next_k_step, num_iter, topk, seed,
                   temporal_split, LOCAL,
                   retrain=True,
                   retrain_checkpoint_idx_to_match=args.retrain_checkpoint_idx_to_match,
                   retrain_str=retrain_str)
    elif training == 2: # unlearn
        # load models to be unlearned

        encoder_pathes = f'./models/encoder_{model_version}_model_best_seed_{seed}.pt'
        decoder_pathes = f'./models/decoder_{model_version}_model_best_seed_{seed}.pt'
        # encoder_pathes = './models/encoder' + str(model_version) + '_model_epoch' + str(model_epoch) + f'_seed_{seed}'
        # decoder_pathes = './models/decoder' + str(model_version) + '_model_epoch' + str(model_epoch) + f'_seed_{seed}'
        encoder_instance = torch.load(encoder_pathes, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        decoder_instance = torch.load(decoder_pathes, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        unlearnIters(history_data, future_data, input_size, encoder_instance, decoder_instance, model_version, training_key_set, val_key_set, retain_key_set, weights,
                   next_k_step, num_iter, topk, seed, temporal_split, LOCAL, user_to_unlearning_items, unlearning_algorithm, args=args)
    
    encs, decs = _find_trained_models(args, model_version, seed)
    if not encs:
        print("No freshly-trained models found – nothing to evaluate.")
    else:
        paired = list(zip(encs, decs))
        _evaluate_and_print(
            paired,
            history_data, future_data,
            input_size,
            val_key_set, test_key_set,
            next_k_step,
            topk_list=[10, 20],          # evaluate at k=10,20 (change as desired)
            temporal_split=temporal_split,
        )

    # else: #test
    #     n = len(training_key_set)
    #     checkpoint_every = math.ceil(n / 4)
    #     checkpoint_idxs = [
    #         i for i in range(n)
    #         if i > 0 and (
    #             (i <= 3 * n // 4 + 5 and i % checkpoint_every == 0)
    #             or (i >= 3 * n // 4 + 5 and i == n - 1)
    #         )
    #     ]
    #     if len(checkpoint_idxs) == 5:                # keep at most 5 checkpoints
    #         checkpoint_idxs = checkpoint_idxs[:4] + [checkpoint_idxs[-1]]

    #     # which checkpoint do we want to mimic?  (if the flag wasn’t supplied, take the last one)
    #     if args.retrain_checkpoint_idx_to_match is None:
    #         unlearning_set_take_first_x = checkpoint_idxs[-1]
    #     else:
    #         unlearning_set_take_first_x = checkpoint_idxs[args.retrain_checkpoint_idx_to_match]

    #     # users whose baskets will be sanitised
    #     users_in_unlearning_set = sorted(training_key_set)[:unlearning_set_take_first_x + 1]

    #     # actually strip the sensitive items; keep track of users rendered unusable (<4 baskets)
    #     filtered_users_in_unlearning_set_no_adaptation = []
    #     for user in users_in_unlearning_set:
    #         unpadded_baskets = history_data[user][1:-1] + [future_data[user][1]]

    #         # remove sensitive items
    #         clean_unpadded_baskets = [
    #             [item for item in basket if item not in user_to_unlearning_items[user]]
    #             for basket in unpadded_baskets
    #         ]
    #         clean_unpadded_baskets = [b for b in clean_unpadded_baskets if len(b) > 0]

    #         if len(clean_unpadded_baskets) < 4:
    #             filtered_users_in_unlearning_set_no_adaptation.append(user)
    #             continue  # don’t keep them in evaluation

    #         # rebuild padded history / future
    #         history_data[user] = [[-1]] + clean_unpadded_baskets[:-1] + [[-1]]
    #         future_data[user]  = [[-1], clean_unpadded_baskets[-1], [-1]]

    #         assert (len(history_data[user]) - 2 + len(future_data[user]) - 2 >= 4), (
    #             f"{user} has insufficient baskets after sanitisation."
    #         )

    #     # remove unusable users everywhere
    #     training_key_set = sorted(set(training_key_set) - set(filtered_users_in_unlearning_set_no_adaptation))
    #     val_key_set = sorted(set(val_key_set) - set(filtered_users_in_unlearning_set_no_adaptation))
    #     test_key_set = sorted(set(test_key_set) - set(filtered_users_in_unlearning_set_no_adaptation))

    #     for i in [10, 20]: #top k
    #         print("Unlearned model")
    #         valid_recall = []
    #         valid_ndcg = []
    #         valid_hr = []
    #         recall_list = []
    #         ndcg_list = []
    #         hr_list = []
    #         print('k = ' + str(i))
    #         # only eval last epoch
    #         for model_epoch in checkpoint_idxs:
    #             print('Users in unlearning set: ', model_epoch)
    #             unlearn_str = f"_sensitive_category_{args.sensitive_category}_unlearning_fraction_{args.unlearning_fraction}"
    #             encoder_pathes = f'./models/unlearn_encoder_{model_version}_model_best_unlearn_epoch{i}_seed_{seed}{unlearn_str}'
    #             decoder_pathes = f'./models/unlearn_decoder_{model_version}_model_best_unlearn_epoch{i}_seed_{seed}{unlearn_str}'
    #             # encoder_pathes = f'./models/encoder_{model_version}_model_epoch{model_epoch}_seed_{seed}'
    #             # decoder_pathes = f'./models/decoder_{model_version}_model_epoch{model_epoch}_seed_{seed}'
    #             if not os.path.exists(encoder_pathes) or not os.path.exists(decoder_pathes):
    #                 print(f"Skip epoch {model_epoch} for unlearning, no new best model or last epoch")
    #                 continue
    #             encoder_instance = torch.load(encoder_pathes, map_location=torch.device('cuda'), weights_only=False)
    #             decoder_instance = torch.load(decoder_pathes, map_location=torch.device('cuda'), weights_only=False)

    #             recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
    #                                         val_key_set, next_k_step, i, test_flag=True, temporal_split=temporal_split)
    #             valid_recall.append(recall)
    #             valid_ndcg.append(ndcg)
    #             valid_hr.append(hr)
    #             recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
    #                                         test_key_set, next_k_step, i, test_flag=True, temporal_split=temporal_split)
    #             recall_list.append(recall)
    #             ndcg_list.append(ndcg)
    #             hr_list.append(hr)
    #         valid_recall = np.asarray(valid_recall)
    #         valid_ndcg = np.asarray(valid_ndcg)
    #         valid_hr = np.asarray(valid_hr)
    #         idx1 = valid_recall.argsort()[::-1][0]
    #         idx2 = valid_ndcg.argsort()[::-1][0]
    #         idx3 = valid_hr.argsort()[::-1][0]
    #         print('max valid recall results:')
    #         print('Epoch: ', idx1)
    #         print('recall: ', recall_list[idx1])
    #         print('ndcg: ', ndcg_list[idx1])
    #         print('phr: ', hr_list[idx1])
    #         sys.stdout.flush()

    #         print('max valid ndcg results:')
    #         print('Epoch: ', idx2)
    #         print('recall: ', recall_list[idx2])
    #         print('ndcg: ', ndcg_list[idx2])
    #         print('phr: ', hr_list[idx2])
    #         sys.stdout.flush()

    #         print('max valid phr results:')
    #         print('Epoch: ', idx3)
    #         print('recall: ', recall_list[idx3])
    #         print('ndcg: ', ndcg_list[idx3])
    #         print('phr: ', hr_list[idx3])
    #         sys.stdout.flush()


    #         print("Initial model:")
    #         valid_recall = []
    #         valid_ndcg = []
    #         valid_hr = []
    #         recall_list = []
    #         ndcg_list = []
    #         hr_list = []
    #         print('k = ' + str(i))
    #         # only eval last epoch
    #         for model_epoch in range(num_iter):
    #             print('Epoch: ', model_epoch)
    #             encoder_pathes = f'./models/encoder_{model_version}_model_epoch{model_epoch}_seed_{seed}'
    #             decoder_pathes = f'./models/decoder_{model_version}_model_epoch{model_epoch}_seed_{seed}'
    #             if not os.path.exists(encoder_pathes) or not os.path.exists(decoder_pathes):
    #                 print(f"Skip epoch {model_epoch}, no new best model or last epoch")
    #                 continue
    #             # encoder_pathes = './models/encoder' + str(model_version) + '_model_epoch' + str(model_epoch) + f'_seed_{seed}'
    #             # decoder_pathes = './models/decoder' + str(model_version) + '_model_epoch' + str(model_epoch) + f'_seed_{seed}'
    #             encoder_instance = torch.load(encoder_pathes, map_location=torch.device('cuda'), weights_only=False)
    #             decoder_instance = torch.load(decoder_pathes, map_location=torch.device('cuda'), weights_only=False)

    #             recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
    #                                         val_key_set, next_k_step, i, test_flag=True, temporal_split=temporal_split)
    #             valid_recall.append(recall)
    #             valid_ndcg.append(ndcg)
    #             valid_hr.append(hr)
    #             recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
    #                                         test_key_set, next_k_step, i, test_flag=True, temporal_split=temporal_split)
    #             recall_list.append(recall)
    #             ndcg_list.append(ndcg)
    #             hr_list.append(hr)
    #         valid_recall = np.asarray(valid_recall)
    #         valid_ndcg = np.asarray(valid_ndcg)
    #         valid_hr = np.asarray(valid_hr)
    #         idx1 = valid_recall.argsort()[::-1][0]
    #         idx2 = valid_ndcg.argsort()[::-1][0]
    #         idx3 = valid_hr.argsort()[::-1][0]
    #         print('max valid recall results:')
    #         print('Epoch: ', idx1)
    #         print('recall: ', recall_list[idx1])
    #         print('ndcg: ', ndcg_list[idx1])
    #         print('phr: ', hr_list[idx1])
    #         sys.stdout.flush()

    #         print('max valid ndcg results:')
    #         print('Epoch: ', idx2)
    #         print('recall: ', recall_list[idx2])
    #         print('ndcg: ', ndcg_list[idx2])
    #         print('phr: ', hr_list[idx2])
    #         sys.stdout.flush()

    #         print('max valid phr results:')
    #         print('Epoch: ', idx3)
    #         print('recall: ', recall_list[idx3])
    #         print('ndcg: ', ndcg_list[idx3])
    #         print('phr: ', hr_list[idx3])
    #         sys.stdout.flush()


    #         print("Retrained models")
    #         valid_recall = []
    #         valid_ndcg = []
    #         valid_hr = []
    #         recall_list = []
    #         ndcg_list = []
    #         hr_list = []
    #         print('k = ' + str(i))
    #         # only eval last epoch
    #         print(f"retrain_checkpoint_idx_to_match: {args.retrain_checkpoint_idx_to_match}")
    #         retrain_str = f"_sensitive_category_{args.sensitive_category}_retrain_checkpoint_idx_to_match_{args.retrain_checkpoint_idx_to_match}"
    #         encoder_pathes = './models/encoder_' + (model_version) + f'_model_best_seed_{seed}{retrain_str}'
    #         decoder_pathes = './models/decoder_' + (model_version) + f'_model_best_seed_{seed}{retrain_str}'
    #         if not os.path.exists(encoder_pathes) or not os.path.exists(decoder_pathes):
    #             print(f"Skip retrain_checkpoint_idx_to_match {args.retrain_checkpoint_idx_to_match}, no model found")
    #             continue
    #         # encoder_pathes = './models/encoder' + str(model_version) + '_model_epoch' + str(model_epoch) + f'_seed_{seed}'
    #         # decoder_pathes = './models/decoder' + str(model_version) + '_model_epoch' + str(model_epoch) + f'_seed_{seed}'
    #         encoder_instance = torch.load(encoder_pathes, map_location=torch.device('cuda'), weights_only=False)
    #         decoder_instance = torch.load(decoder_pathes, map_location=torch.device('cuda'), weights_only=False)

    #         recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
    #                                     val_key_set, next_k_step, i, test_flag=True, temporal_split=temporal_split)
    #         valid_recall.append(recall)
    #         valid_ndcg.append(ndcg)
    #         valid_hr.append(hr)
    #         recall, ndcg, hr = evaluate(history_data, future_data, encoder_instance, decoder_instance, input_size,
    #                                     test_key_set, next_k_step, i, test_flag=True, temporal_split=temporal_split)
    #         recall_list.append(recall)
    #         ndcg_list.append(ndcg)
    #         hr_list.append(hr)
    #         valid_recall = np.asarray(valid_recall)
    #         valid_ndcg = np.asarray(valid_ndcg)
    #         valid_hr = np.asarray(valid_hr)
    #         idx1 = valid_recall.argsort()[::-1][0]
    #         idx2 = valid_ndcg.argsort()[::-1][0]
    #         idx3 = valid_hr.argsort()[::-1][0]
    #         print('max valid recall results:')
    #         print('Epoch: ', idx1)
    #         print('recall: ', recall_list[idx1])
    #         print('ndcg: ', ndcg_list[idx1])
    #         print('phr: ', hr_list[idx1])
    #         sys.stdout.flush()

    #         print('max valid ndcg results:')
    #         print('Epoch: ', idx2)
    #         print('recall: ', recall_list[idx2])
    #         print('ndcg: ', ndcg_list[idx2])
    #         print('phr: ', hr_list[idx2])
    #         sys.stdout.flush()

    #         print('max valid phr results:')
    #         print('Epoch: ', idx3)
    #         print('recall: ', recall_list[idx3])
    #         print('ndcg: ', ndcg_list[idx3])
    #         print('phr: ', hr_list[idx3])
    #         sys.stdout.flush()



if __name__ == "__main__":
    unlearn_main()