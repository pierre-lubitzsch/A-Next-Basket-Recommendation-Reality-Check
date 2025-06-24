import pickle
import math
import os
import torch
from torch.nn.utils import parameters_to_vector
import csv
import sys


from sets2sets_new import EncoderRNN_new, AttnDecoderRNN_new


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
    use_cuda = False
    seeds = [2, 3, 5, 7, 11]
    categories = ["baby", "alcohol", "meat"]
    datasets = ["Instacart"]
    unlearning_fractions = [0.001]
    unlearning_algorithms = ["scif", "fanchuan", "kookmin"]

    results = []

    directory = "./models"
    for filename in sorted(os.listdir(directory)):
        if "decoder" in filename or "unlearn" not in filename or "unlearning_fraction_0.001" not in filename:
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

        param_distance_unlearned_retrained = coupled_distance(unlearned_encoder, unlearned_decoder, retrained_encoder, retrained_decoder)
        param_distance_original_retrained = coupled_distance(original_encoder, original_decoder, retrained_encoder, retrained_decoder)
        param_distance_unlearned_original = coupled_distance(unlearned_encoder, unlearned_decoder, original_encoder, original_decoder)

        results.append([encoder_filename, retrain_encoder_filename, original_encoder_filename, param_distance_unlearned_retrained, param_distance_original_retrained, param_distance_unlearned_original])
        print(f"Unlearned encoder: {encoder_filename}")
        print(f"Retrained encoder: {retrain_encoder_filename}")
        print(f"Original encoder: {original_encoder_filename}")
        print(f"Parameter distance unlearned vs retrained: {param_distance_unlearned_retrained}\n")
        print(f"Parameter distance original vs retrained: {param_distance_original_retrained}")
        print(f"Parameter distance unlearned vs original: {param_distance_unlearned_original}")
        sys.stdout.flush()


    out_file = f"{directory}/sets2sets_param_distances.csv"
    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["unlearned_encoder", "retrained_encoder", "original encoder", "unlearned_vs_retrained_mse", "original_vs_retrained_mse", "unlearned_vs_original_mse"])
        writer.writerows(results)