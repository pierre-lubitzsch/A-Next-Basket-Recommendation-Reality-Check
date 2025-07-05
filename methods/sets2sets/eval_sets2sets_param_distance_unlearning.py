import pickle
import math
import os
import torch
from torch.nn.utils import parameters_to_vector
import csv
import sys
import json

from sets2sets_new import EncoderRNN_new, AttnDecoderRNN_new, decoding_next_k_step


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


if __name__ == "__main__":
    use_cuda = True
    seeds = [2, 3, 5, 7, 11]
    categories = ["baby", "alcohol", "meat"]
    datasets = ["Instacart"]
    unlearning_fractions = [0.001]
    unlearning_algorithms = ["scif", "fanchuan", "kookmin"]
    topk_list = [10, 20]
    next_k_step = 1

    results = []
    filenames_seen = set()
    
    device = torch.device('cuda' if use_cuda else 'cpu')

    directory = "./models"
    for filename in sorted(os.listdir(directory)):
        if "decoder" in filename or ("unlearn_epoch" not in filename):
            continue
        
        encoder_filename = filename
        decoder_filename = filename.replace("encoder", "decoder")

        encoder_path = f"{directory}/{encoder_filename}"
        decoder_path = f"{directory}/{decoder_filename}"
        
        retrain_encoder_filename, retrain_decoder_filename, original_encoder_filename, original_decoder_filename = unlearn_model_to_retrained_model(filename)
        if retrain_encoder_filename is None or retrain_decoder_filename is None:
            continue

        retrain_encoder_path = f"{directory}/{retrain_encoder_filename}"
        retrain_decoder_path = f"{directory}/{retrain_decoder_filename}"

        original_encoder_path = f"{directory}/{original_encoder_filename}"
        original_decoder_path = f"{directory}/{original_decoder_filename}"

        original_encoder = torch.load(original_encoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        original_decoder = torch.load(original_decoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        unlearned_encoder = torch.load(encoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        unlearned_decoder = torch.load(decoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        retrained_encoder = torch.load(retrain_encoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)
        retrained_decoder = torch.load(retrain_decoder_path, map_location=torch.device('cuda' if use_cuda else 'cpu'), weights_only=False)

        print(f"Processing file: {filename}")
        sys.stdout.flush()

        param_distance_unlearned_retrained = coupled_distance(unlearned_encoder, unlearned_decoder, retrained_encoder, retrained_decoder)
        param_distance_original_retrained = coupled_distance(original_encoder, original_decoder, retrained_encoder, retrained_decoder)
        param_distance_unlearned_original = coupled_distance(unlearned_encoder, unlearned_decoder, original_encoder, original_decoder)

        print(f"Unlearned encoder: {encoder_filename}")
        print(f"Retrained encoder: {retrain_encoder_filename}")
        print(f"Original encoder: {original_encoder_filename}")
        print(f"Parameter distance unlearned vs retrained: {param_distance_unlearned_retrained}")
        print(f"Parameter distance original vs retrained: {param_distance_original_retrained}")
        print(f"Parameter distance unlearned vs original: {param_distance_unlearned_original}\n\n")
        sys.stdout.flush()

        del original_encoder, original_decoder, unlearned_encoder, unlearned_decoder, retrained_encoder, retrained_decoder

        history_file = "../../jsondata/instacart_history.json"
        future_file = "../../jsondata/instacart_future.json"
        keyset_file = "../../keyset/instacart_keyset_0.json"
        unlearning_data_file = f"../../unlearning_data/dataset_instacart_seed_{filename.split('_seed_')[-1].split('_')[0]}_method_sensitive_unlearning_fraction_0.001.pkl"

        with open(history_file, 'r') as f:
            history_data = json.load(f)
        with open(future_file, 'r') as f:
            future_data = json.load(f)
        with open(keyset_file, 'r') as f:
            keyset = json.load(f)
            input_size = keyset['item_num']
        with open(unlearning_data_file, "rb") as f:
            user_to_unlearning_items = pickle.load(f)
            sensitive_category = filename.split("sensitive_category_")[-1].split("_")[0]
            user_to_unlearning_items = user_to_unlearning_items[sensitive_category]

        # sensitive item prediction:
        for cur_encoder_filename, cur_decoder_filename in [(encoder_filename, decoder_filename), (retrain_encoder_filename, retrain_decoder_filename), (original_encoder_filename, original_decoder_filename)]:
            if cur_encoder_filename in filenames_seen:
                continue

            cur_encoder_filepath = f"{directory}/{cur_encoder_filename}"
            cur_decoder_filepath = f"{directory}/{cur_decoder_filename}"

            encoder = torch.load(cur_encoder_filepath, map_location=device, weights_only=False)
            decoder = torch.load(cur_decoder_filepath, map_location=device, weights_only=False)
            print(f"sensitive item prediction for: {cur_encoder_filename}")
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

            with torch.no_grad():
                for k in topk_list:
                    print(f"k = {k}")            
                    sensitive_item_in_output_basket_count = 0
                    # sensitive item prediction:
                    for user in user_to_unlearning_items:
                        # training_pair = training_pairs[iter - 1]
                        # input_variable = training_pair[0]
                        # target_variable = training_pair[1]
                        
                        unpadded_baskets = history_data[user][1:-1] + [future_data[user][1]]
                        clean_unpadded_baskets = [[item for item in basket if item not in user_to_unlearning_items[user]] for basket in unpadded_baskets]
                        clean_unpadded_baskets = list(filter(lambda x: len(x) > 0, clean_unpadded_baskets))
                        if len(clean_unpadded_baskets) < 4:
                            continue
                        
                        target_variable = [[-1], clean_unpadded_baskets[1], [-1]]
                        input_variable = [[-1]] + clean_unpadded_baskets[:-1] + [[-1]]


                        output_vectors, prob_vectors = decoding_next_k_step(encoder, decoder, input_variable, target_variable,
                                                                            input_size, next_k_step, k)
                        
                        predicted_basket = output_vectors[0]
                        predicted_basket_ints_set = set(int(t.item()) for t in predicted_basket)

                        sensitive_items_predicted = predicted_basket_ints_set & set(user_to_unlearning_items[user])
                        sensitive_item_in_output_basket_count += int(len(sensitive_items_predicted) > 0)
                    
                    print(f"{sensitive_item_in_output_basket_count}/{len(cur_user_to_unlearning_items)} users have sensitive items in their output basket")
                    results.append([encoder_filename, retrain_encoder_filename, original_encoder_filename, param_distance_unlearned_retrained, param_distance_original_retrained, param_distance_unlearned_original, k, cur_encoder_filename, sensitive_item_in_output_basket_count])

        filenames_seen |= set([encoder_filename, retrain_encoder_filename, original_encoder_filename])
        print("\n\n")
    

    out_file = f"{directory}/sets2sets_unlearning_sensitive_evaluation.csv"
    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["unlearned_encoder", "retrained_encoder", "original encoder", "unlearned_vs_retrained_mse", "original_vs_retrained_mse", "unlearned_vs_original_mse", "k", "sensitive_encoder_filename", "sensitive_item_in_output_basket_count"])
        writer.writerows(results)
