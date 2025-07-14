import pickle
import math
import os
import torch
from torch.nn.utils import parameters_to_vector
import csv
import sys
import json
import pandas as pd
import argparse
import numpy as np
import torch

from sets2sets_new import EncoderRNN_new, AttnDecoderRNN_new, decoding_next_k_step, get_precision_recall_Fscore, get_HT, get_NDCG


def unlearn_model_to_retrained_model(unlearn_filename):
    ds = "Instacart"
    seed = int(unlearn_filename.split("_seed_")[-1].split("_")[0])
    frac = float(unlearn_filename.split("_unlearning_fraction_")[-1].split("_")[0])
    if "unlearn_epoch" not in unlearn_filename:
        return None, None, None, None
    unlearn_epochs = int(unlearn_filename.split("unlearn_epoch")[-1].split("_")[0])
    if unlearn_epochs < 10:
        return None, None, None, None
    if "sensitive" in unlearn_filename:
        category = unlearn_filename.split("sensitive_category_")[-1].split("_")[0]
    
    with open(f"../../unlearning_data/dataset_{ds.lower()}_seed_{seed}_method_sensitive_unlearning_fraction_{frac}.pkl", "rb") as f:
        unlearn_users_to_items = pickle.load(f)
        if "sensitive" in unlearn_filename:
            unlearn_users_to_items = unlearn_users_to_items[category]


    n = len(unlearn_users_to_items)
    checkpoint_every = math.ceil(n / 4)
    checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]
    if len(checkpoint_idxs) == 5:
        checkpoint_idxs = checkpoint_idxs[:4] + [checkpoint_idxs[-1]]

    retrain_idx_to_match = checkpoint_idxs.index(unlearn_epochs)
    if retrain_idx_to_match == -1:
        return None, None, None, None
    
    encoder_retrain_filename = f"encoder_instacart0_model_best_seed_{seed}_sensitive_category_{category}_unlearning_fraction_{frac}_retrain_checkpoint_idx_to_match_{retrain_idx_to_match}.pt"
    decoder_retrain_filename = f"decoder_instacart0_model_best_seed_{seed}_sensitive_category_{category}_unlearning_fraction_{frac}_retrain_checkpoint_idx_to_match_{retrain_idx_to_match}.pt"
    
    original_encoder_filename = f"encoder_instacart0_model_best_seed_{seed}.pt"
    original_decoder_filename = f"decoder_instacart0_model_best_seed_{seed}.pt"
    
    return encoder_retrain_filename, decoder_retrain_filename, original_encoder_filename, original_decoder_filename



def coupled_distance(enc_a, dec_a, enc_b, dec_b, device="cpu"):
    """
    p-norm of the *combined* parameter vector of (encoder, decoder).

    enc_a / dec_a : un-/pre-trained pair
    enc_b / dec_b : retrained pair
    """
    with torch.no_grad():
        vec_a = torch.cat([
            parameters_to_vector([p.detach().to(device) for p in enc_a.parameters()]),
            parameters_to_vector([p.detach().to(device) for p in dec_a.parameters()])
        ])

        vec_b = torch.cat([
            parameters_to_vector([p.detach().to(device) for p in enc_b.parameters()]),
            parameters_to_vector([p.detach().to(device) for p in dec_b.parameters()])
        ])

    param_diff = vec_a - vec_b
    # MSE
    return (param_diff ** 2).mean().item()


def main(args):
    use_cuda = True
    seeds = [2, 3, 5, 7, 11]
    categories = ["baby", "alcohol", "meat"]
    datasets = ["Instacart"]
    unlearning_fractions = [0.001]
    unlearning_algorithms = ["scif", "fanchuan", "kookmin"]
    topk_list = [10, 20]
    next_k_step = 1

    history_file = "../../jsondata/instacart_history.json"
    future_file = "../../jsondata/instacart_future.json"
    keyset_file = "../../keyset/instacart_keyset_0.json"

    category_to_aisles = {
        "meat": [5, 95, 96, 15, 33, 34, 35, 49, 106, 122],
        "alcohol": [27, 28, 62, 124, 134],
        "baby": [82, 92, 102, 56],
    }

    products_with_aisle_id_filepath = f"../../dataset/instacart_products.csv"
    products = pd.read_csv(products_with_aisle_id_filepath)

    category_to_items = {
        cat: set(products[products["aisle_id"].isin(aisle_ids)]["product_id"])
        for cat, aisle_ids in category_to_aisles.items()
    }

    with open(history_file, 'r') as f:
        history_data = json.load(f)
    with open(future_file, 'r') as f:
        future_data = json.load(f)
    with open(keyset_file, 'r') as f:
        keyset = json.load(f)
        input_size = keyset['item_num']
    stats_from_log = pd.read_csv("./sets2sets_combined_results.csv")

    results = []
    filenames_seen = set()
    
    device = torch.device('cuda' if use_cuda else 'cpu')

    directory = "./models"

    for filename in sorted(os.listdir(directory)):
        if "decoder" in filename or ("unlearn_epoch" not in filename) or (args.category != "all" and f"category_{args.category}" not in filename):
            continue
        
        encoder_filename = filename
        decoder_filename = filename.replace("encoder", "decoder")

        encoder_path = f"{directory}/{encoder_filename}"
        decoder_path = f"{directory}/{decoder_filename}"

        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            continue
        
        retrain_encoder_filename, retrain_decoder_filename, original_encoder_filename, original_decoder_filename = unlearn_model_to_retrained_model(filename)
        if retrain_encoder_filename is None or retrain_decoder_filename is None:
            continue

        retrain_encoder_path = f"{directory}/{retrain_encoder_filename}"
        retrain_decoder_path = f"{directory}/{retrain_decoder_filename}"

        original_encoder_path = f"{directory}/{original_encoder_filename}"
        original_decoder_path = f"{directory}/{original_decoder_filename}"

        # original_encoder = torch.load(original_encoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        # original_decoder = torch.load(original_decoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        # unlearned_encoder = torch.load(encoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        # unlearned_decoder = torch.load(decoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        # retrained_encoder = torch.load(retrain_encoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        # retrained_decoder = torch.load(retrain_decoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        # print(f"Processing file: {encoder_filename}")

        # param_distance_unlearned_retrained = coupled_distance(unlearned_encoder, unlearned_decoder, retrained_encoder, retrained_decoder)
        # param_distance_original_retrained = coupled_distance(original_encoder, original_decoder, retrained_encoder, retrained_decoder)
        # param_distance_unlearned_original = coupled_distance(unlearned_encoder, unlearned_decoder, original_encoder, original_decoder)

        # print(f"Unlearned encoder: {encoder_filename}")
        # print(f"Retrained encoder: {retrain_encoder_filename}")
        # print(f"Original encoder: {original_encoder_filename}")
        # print(f"Parameter distance unlearned vs retrained: {param_distance_unlearned_retrained}")
        # print(f"Parameter distance original vs retrained: {param_distance_original_retrained}")
        # print(f"Parameter distance unlearned vs original: {param_distance_unlearned_original}\n\n")

        # del original_encoder, original_decoder, unlearned_encoder, unlearned_decoder, retrained_encoder, retrained_decoder

        
        unlearning_data_file = f"../../unlearning_data/dataset_instacart_seed_{filename.split('_seed_')[-1].split('_')[0]}_method_sensitive_unlearning_fraction_0.001.pkl"

        with open(unlearning_data_file, "rb") as f:
            user_to_unlearning_items = pickle.load(f)
            sensitive_category = filename.split("sensitive_category_")[-1].split("_")[0]
            user_to_unlearning_items = user_to_unlearning_items[sensitive_category]

        user_list = list(future_data.keys())

        for cur_encoder_filename, cur_decoder_filename in [(encoder_filename, decoder_filename), (retrain_encoder_filename, retrain_decoder_filename)]:#, (original_encoder_filename, original_decoder_filename)]:
            if cur_encoder_filename in filenames_seen:
                continue

            compare_to_retrain = "retrain" not in cur_encoder_filename
            if compare_to_retrain:
                retrained_encoder = torch.load(retrain_encoder_path, map_location=device, weights_only=False)
                retrained_decoder = torch.load(retrain_decoder_path, map_location=device, weights_only=False)
                retrained_encoder.eval()
                retrained_decoder.eval()

            cur_encoder_filepath = f"{directory}/{cur_encoder_filename}"
            cur_decoder_filepath = f"{directory}/{cur_decoder_filename}"

            if not os.path.exists(cur_encoder_filepath) or not os.path.exists(cur_decoder_filepath):
                continue

            encoder = torch.load(cur_encoder_filepath, map_location=device, weights_only=False)
            decoder = torch.load(cur_decoder_filepath, map_location=device, weights_only=False)
            encoder.eval()
            decoder.eval()

            cur_user_to_unlearning_items = user_to_unlearning_items
            users_to_take = len(cur_user_to_unlearning_items)
            if "unlearn_epoch" in cur_encoder_filename:
                users_to_take = int(cur_encoder_filename.split("unlearn_epoch")[-1].split("_")[0]) + 1
                users = set(sorted(cur_user_to_unlearning_items.keys())[:users_to_take])
                cur_user_to_unlearning_items = {user: cur_user_to_unlearning_items[user] for user in users if user in cur_user_to_unlearning_items}
            elif "retrain_checkpoint_idx_to_match" in cur_encoder_filename:
                n = len(cur_user_to_unlearning_items)
                checkpoint_every = (n + 3) // 4 # ceil
                checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]
                idx = int(cur_encoder_filename.split("retrain_checkpoint_idx_to_match_")[-1].split(".")[0])
                users_to_take = checkpoint_idxs[idx] + 1
                users = set(sorted(cur_user_to_unlearning_items.keys())[:users_to_take])
                cur_user_to_unlearning_items = {user: cur_user_to_unlearning_items[user] for user in users if user in cur_user_to_unlearning_items}


            cur_seed = int(cur_encoder_filename.split("seed_")[-1].split("_")[0])
            cur_category = sensitive_category
            cur_requests = round(100 * users_to_take / len(user_to_unlearning_items))

            # account for rounding errors
            candidates = (25, 50, 75, 100)
            cur_requests = min(candidates, key=lambda x: abs(x - cur_requests))


            cur_algorithm = "Baseline"

            if "unlearn_epoch" not in cur_encoder_filename:
                cur_algorithm = "Retrain"
            else:
                unlearning_algorithm_names = ["Fanchuan", "Kookmin", "SCIF"]
                for unlearn_algo_name in unlearning_algorithm_names:
                    if unlearn_algo_name.lower() in cur_encoder_filename:
                        cur_algorithm = unlearn_algo_name
                        break
            
            cur_time_elapsed = stats_from_log[
                (stats_from_log["seed"] == cur_seed)
                    & (stats_from_log["algorithm"] == cur_algorithm.lower())
                    & (stats_from_log["category"] == cur_category.lower())
                    & (stats_from_log["Frac"] == f"{round(cur_requests * 4 / 100)}/4")
            ]["elapsed"]

            # training did not converge in this case
            if len(cur_time_elapsed) == 0:
                continue


            performance_metrics_rnh = []
            sensitive_item_percentages = []
            kl_div_list = []
            js_div_list = []

            with torch.no_grad():
                for k_idx, k in enumerate(topk_list):
                    prec = []
                    rec = []
                    F = []
                    prec1 = []
                    rec1 = []
                    F1 = []
                    prec2 = []
                    rec2 = []
                    F2 = []
                    prec3 = []
                    rec3 = []
                    F3 = []

                    NDCG = []
                    n_hit = 0
                    count = 0


                    print(f"k = {k}")            
                    sensitive_item_in_output_basket_count = 0
                    # sensitive item prediction:
                    for user in user_list:#cur_user_to_unlearning_items:
                        # training_pair = training_pairs[iter - 1]
                        # input_variable = training_pair[0]
                        # target_variable = training_pair[1]
                        
                        unpadded_baskets = history_data[user][1:-1] + [future_data[user][1]]
                        if user in cur_user_to_unlearning_items:
                            clean_unpadded_baskets = [[item for item in basket if item not in category_to_items[sensitive_category]] for basket in unpadded_baskets]
                            clean_unpadded_baskets = list(filter(lambda x: len(x) > 0, clean_unpadded_baskets))
                            if len(clean_unpadded_baskets) < 4:
                                continue
                        else:
                            clean_unpadded_baskets = unpadded_baskets
                            clean_unpadded_baskets = list(filter(lambda x: len(x) > 0, clean_unpadded_baskets))
                            if len(clean_unpadded_baskets) < 4:
                                continue
                        
                        target_variable = [[-1], clean_unpadded_baskets[-1], [-1]]
                        input_variable = [[-1]] + clean_unpadded_baskets[:-1] + [[-1]]


                        output_vectors, prob_vectors, decoder_output = decoding_next_k_step(encoder, decoder, input_variable, target_variable,
                                                                            input_size, next_k_step, k, return_decoder_output=True)
                        
                        # sensitive item prediction
                        if user in cur_user_to_unlearning_items:
                            predicted_basket = output_vectors[0]
                            predicted_basket_ints_set = set(int(t.item()) for t in predicted_basket)

                            sensitive_items_predicted = predicted_basket_ints_set & set(category_to_items[sensitive_category])
                            sensitive_item_in_output_basket_count += int(len(sensitive_items_predicted) > 0)
                        
                        # KL-divergenge KL(retrained || unlearned) (k_idx == 0 to only do it once, also only for users who submitted forget requests)
                        if compare_to_retrain and user in cur_user_to_unlearning_items and k_idx == 0:
                            _, _, retrain_decoder_output = decoding_next_k_step(retrained_encoder, retrained_decoder, input_variable, target_variable,
                                                                            input_size, next_k_step, k, return_decoder_output=True)
                            
                            p_unlearned = decoder_output
                            p_retrained = retrain_decoder_output
                            mid = (p_unlearned + p_retrained) / 2

                            # eps for numeric stability
                            eps = 1e-20
                            log_p_unlearned = torch.log(p_unlearned + eps)
                            log_mid = torch.log(mid + eps)
                            kl = torch.nn.functional.kl_div(log_p_unlearned, p_retrained, reduction='batchmean')
                            kl_div_list.append(kl.cpu().item())

                            kl_p_m = torch.nn.functional.kl_div(log_mid, p_retrained, reduction='batchmean')
                            kl_q_m = torch.nn.functional.kl_div(log_mid, p_unlearned, reduction='batchmean')
                            js = (kl_p_m + kl_q_m) / 2
                            js_div_list.append(js.cpu().item())

                        # performance metrics calculated for everything
                        output_size = input_size
                        hit = 0
                        for idx in range(len(output_vectors)):
                            # for idx in [2]:
                            vectorized_target = np.zeros(output_size)
                            for ii in target_variable[1 + idx]: #target_variable[[-1], [item, item], .., [-1]]
                                vectorized_target[ii] = 1

                            vectorized_output = np.zeros(output_size)
                            for ii in output_vectors[idx]:
                                vectorized_output[ii] = 1

                            precision, recall, Fscore, correct = get_precision_recall_Fscore(vectorized_target, vectorized_output)
                            prec.append(precision)
                            rec.append(recall)
                            F.append(Fscore)
                            if idx == 0:
                                prec1.append(precision)
                                rec1.append(recall)
                    
                                rec2.append(recall)
                                F2.append(Fscore)
                            elif idx == 2:
                                prec3.append(precision)
                                rec3.append(recall)
                                F3.append(Fscore)
                            # length[idx] += np.sum(target_variable[1 + idx])
                            # prob_vectors is the probability
                            target_topi = prob_vectors[idx]
                            hit += get_HT(vectorized_target, target_topi, k)
                            ndcg = get_NDCG(vectorized_target, target_topi, k)
                            NDCG.append(ndcg)
                        if hit == next_k_step:
                            n_hit += 1

                    recall, ndcg, hitrate = np.mean(rec), np.mean(NDCG), n_hit / len(user_list)

                    # results.append([encoder_filename, retrain_encoder_filename, original_encoder_filename, param_distance_unlearned_retrained, param_distance_original_retrained, param_distance_unlearned_original, k, cur_encoder_filename, sensitive_item_in_output_basket_count])
                    performance_metrics_rnh.append((recall, ndcg, hitrate))
                    cur_sensitive_item_percentage = sensitive_item_in_output_basket_count / len(cur_user_to_unlearning_items)
                    sensitive_item_percentages.append(100 * cur_sensitive_item_percentage)


                cur_category = sensitive_category
                cur_requests = round(100 * users_to_take / len(user_to_unlearning_items))

                # account for rounding errorr
                candidates = (25, 50, 75, 100)
                cur_requests = min(candidates, key=lambda x: abs(x - cur_requests))

                cur_algorithm = "Baseline"
                if "unlearn_epoch" not in cur_encoder_filename:
                    cur_algorithm = "Retrain"
                else:
                    unlearning_algorithm_names = ["Fanchuan", "Kookmin", "SCIF"]
                    for unlearn_algo_name in unlearning_algorithm_names:
                        if unlearn_algo_name.lower() in cur_encoder_filename:
                            cur_algorithm = unlearn_algo_name
                            break

                cur_recall_10, cur_ndcg_10, cur_hitrate_10 = performance_metrics_rnh[0]
                cur_recall_20, cur_ndcg_20, cur_hitrate_20 = performance_metrics_rnh[1]
                cur_seed = int(cur_encoder_filename.split("seed_")[-1].split("_")[0])
                # load elapsed time from log file
                cur_time_elapsed = stats_from_log[
                    (stats_from_log["seed"] == cur_seed)
                     & (stats_from_log["algorithm"] == cur_algorithm.lower())
                     & (stats_from_log["category"] == cur_category.lower())
                     & (stats_from_log["Frac"] == f"{round(cur_requests * 4 / 100)}/4")]["elapsed"].values[0]
                cur_time = cur_time_elapsed / len(cur_user_to_unlearning_items) if len(cur_user_to_unlearning_items) > 0 else 0
                cur_kl_div = np.mean(kl_div_list) if compare_to_retrain else 0
                cur_js_div = np.mean(js_div_list) if compare_to_retrain else 0
                cur_sensitive_items_10, cur_sensitive_items_20 = sensitive_item_percentages

                results.append((
                    cur_category,
                    cur_requests,
                    cur_algorithm,
                    cur_recall_10,
                    cur_ndcg_10,
                    cur_hitrate_10,
                    cur_recall_20,
                    cur_ndcg_20,
                    cur_hitrate_20,
                    cur_seed,
                    cur_time,
                    cur_sensitive_items_10,
                    cur_sensitive_items_20,
                    cur_kl_div,
                    cur_js_div,
                ))

                print("Appended result:\n" \
                    f"  Category:               {cur_category}\n" \
                    f"  Requests:               {cur_requests}\n" \
                    f"  Algorithm:              {cur_algorithm}\n" \
                    f"  Rec@10:              {cur_recall_10:.4f}\n" \
                    f"  nDCG@10:                {cur_ndcg_10:.4f}\n" \
                    f"  PHR@10:                 {cur_hitrate_10:.4f}\n" \
                    f"  Rec@20:              {cur_recall_20:.4f}\n" \
                    f"  nDCG@20:                {cur_ndcg_20:.4f}\n" \
                    f"  PHR@20:                 {cur_hitrate_20:.4f}\n" \
                    f"  Seed:                   {cur_seed}\n" \
                    f"  Time per request (s):   {cur_time:.4f}\n" \
                    f"  Sensitive items @10:    {cur_sensitive_items_10}\n" \
                    f"  Sensitive items @20:    {cur_sensitive_items_20}\n" \
                    f"  KL(Retrained || Unlearned): {cur_kl_div:.6f}\n" \
                    f"  JS(Retrained || Unlearned): {cur_js_div:.6f}\n\n"
                )

        filenames_seen |= set([encoder_filename, retrain_encoder_filename])
        print("\n\n")
        sys.stdout.flush()
    

    out_file_per_seed = f"{directory}/sets2sets_unlearning_sensitive_evaluation_per_seed_category_{args.category}.csv"
    columns = [
        "Category",
        "Requests",
        "Algorithm",
        "Rec@10",
        "nDCG@10",
        "PHR@10",
        "Rec@20",
        "nDCG@20",
        "PHR@20",
        "seed",
        "Time per request (s)",
        "Sensitive items (10)",
        "Sensitive items (20)",
        "KL(Retrained || Unlearned)",
        "JS(Retrained || Unlearned)",
    ]

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(out_file_per_seed, index=False)

    out_file_averaged_over_seeds = f"{directory}/sets2sets_unlearning_sensitive_evaluation_averaged_over_seeds_category_{args.category}.csv"
    group_cols = ["Category", "Requests", "Algorithm"]
    avg_cols = [
        "Rec@10", "nDCG@10", "PHR@10",
        "Rec@20", "nDCG@20", "PHR@20",
        "Time per request (s)",
        "Sensitive items (10)",
        "Sensitive items (20)",
        "KL(Retrained || Unlearned)",
        "JS(Retrained || Unlearned)",
    ]

    df_avg = (
        df
        .groupby(group_cols, as_index=False)[avg_cols]
        .mean()
    )

    df_avg.to_csv(out_file_averaged_over_seeds, index=False)
    print(f"Saved raw evaluation results to {out_file_per_seed} and results averaged over seeds to {out_file_averaged_over_seeds}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Sets2Sets unlearning metrics for a single sensitive category"
    )
    parser.add_argument(
        "-c",
        "--category",
        default="all",
        help="Sensitive category to process (e.g. baby, alcohol, meat). A value of 'all' will process all categories.",
        choices=["all", "baby", "alcohol", "meat"],
    )
    args = parser.parse_args()
    main(args)