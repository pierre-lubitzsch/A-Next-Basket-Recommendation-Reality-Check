import sys

sys.path.append("..")
DEBUG = False
if DEBUG:
    sys.path.append("./methods/dnntsp")
    sys.path.append("./methods/dnntsp/train")

from train_model import train_model
from model.temporal_set_prediction import temporal_set_prediction
from utils.util import get_class_weights
from utils.loss import BPRLoss, WeightMSELoss
from utils.data_container import get_data_loader, get_data_loader_temporal_split
from utils.load_config import get_attribute
import torch
import torch.nn as nn
import os
import shutil
import argparse
from train_main import evaluate_best_model
from utils.util import save_model, convert_to_gpu, convert_all_data_to_gpu, load_model

import pickle
import math
import json
import random
import numpy as np
import time

import scif
import kookmin
import fanchuan


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    parser = argparse.ArgumentParser(description='Training script arguments')
    parser.add_argument('--save_model_folder', type=str, default='DNNTSP', help='Folder to save the model')
    parser.add_argument('--history_path', type=str, default='../../../jsondata/tafeng_history.json', help='Path to history JSON file')
    parser.add_argument('--future_path', type=str, default='../../../jsondata/tafeng_future.json', help='Path to future JSON file')
    parser.add_argument('--keyset_path', type=str, default='../../../keyset/tafeng_keyset_0.json', help='Path to keyset JSON file')
    parser.add_argument('--item_embed_dim', type=int, default=32, help='Dimension of item embeddings')
    parser.add_argument('--loss_function', type=str, default='multi_label_soft_loss', help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty)')
    parser.add_argument('--data_path', type=str, default='../../../jsondata/tafeng_history.json', help='Data path for class weights')
    parser.add_argument('--LOCAL', action='store_true', help='Set this flag to run locally')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--temporal_split', action="store_true", help='set this flag if you want a temporal split instead of a user split')
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
            "fanchuan",
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


def create_model(save_model_folder, item_embed_dim):
    data = get_attribute("data")
    items_total = get_attribute("items_total")
    print(f"Using model settings: {data}/{save_model_folder}")
    model = temporal_set_prediction(items_total=items_total,
                                    item_embedding_dim=item_embed_dim)
    return model


def create_loss(loss_function, data_path):
    if loss_function == 'bpr_loss':
        loss_func = BPRLoss()
    elif loss_function == 'mse_loss':
        loss_func = WeightMSELoss()
    elif loss_function == 'weight_mse_loss':
        loss_func = WeightMSELoss(weights=get_class_weights(data_path))
    elif loss_function == "multi_label_soft_loss":
        loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
    else:
        raise ValueError("Unknown loss function.")
    return loss_func


def unlearn(save_model_folder, history_path, future_path, keyset_path, item_embed_dim, loss_function, epochs, batch_size, learning_rate, optim, weight_decay, data_path, LOCAL=False, temporal_split=False, args=None,):
    script_start = time.perf_counter()
    
    loss_func = create_loss(loss_function, data_path)

    data = get_attribute("data")
    
    if LOCAL:
        model_folder = f"../save_model_folder/{data}/{save_model_folder}"
        tensorboard_folder = f"../results/runs/{data}/{save_model_folder}"
    else:
        model_folder = f"/opt/results/save_model_folder/{data}/{save_model_folder}"
        tensorboard_folder = f"/opt/results/runs/{data}/{save_model_folder}"

    model = create_model(save_model_folder, item_embed_dim)
    best_model_path = f'{model_folder}/model_best_seed_{args.seed}.pkl'
    model = load_model(model, best_model_path)
    # TODO: save best models with this name on cluster to get

    # shutil.rmtree(model_folder, ignore_errors=True)
    os.makedirs(model_folder, exist_ok=True)
    # shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder, exist_ok=True)

    if args.unlearning_algorithm == "kookmin":
        # params which were not initialized have a lower gradient.
        # as these are the majority we scale down the lr globally and locally scale it up again for the reinitialized params
        learning_rate *= 0.1
    if optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    elif optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9)
    else:
        raise NotImplementedError("The specified optimizer is not implemented.")


    model = convert_to_gpu(model)
    model.train()
    loss_func = convert_to_gpu(loss_func)

    data = get_attribute("data")
    percentage_str = f"_popular_percentage_{args.popular_percentage}" if args.method in ["popular", "unpopular"] else ""
    fraction_str = f"_unlearning_fraction_{args.unlearning_fraction}" if args.method in ["popular", "unpopular", "random", "sensitive"] else ""
    unlearning_data_file = f'../../../unlearning_data/dataset_{data.lower()}_seed_{args.seed}_method_{args.method}{fraction_str}{percentage_str}.pkl'

    with open(unlearning_data_file, "rb") as f:
        user_to_unlearning_items = pickle.load(f)
        if args.method == "sensitive":
            user_to_unlearning_items = user_to_unlearning_items[args.sensitive_category]

    n = len(user_to_unlearning_items)
    checkpoint_every = math.ceil(n / 4)
    checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]

    unlearning_user_ids = sorted(user_to_unlearning_items.keys())

    # for kookmin
    param_list = [p for p in model.parameters() if p.requires_grad]
    param_index = {id(p): i for i,p in enumerate(param_list)}

    with open(future_path, 'r') as f:
        data_future = json.load(f)
    
    user_list = sorted(data_future.keys(), key=int)
    unlearning_user_set = set(unlearning_user_ids)
    retain_user_ids = [u for u in user_list if u not in unlearning_user_set]

    with open(history_path, 'r') as f:
        data_history = json.load(f)
    with open(future_path, 'r') as f:
        data_future  = json.load(f)

    # if you're doing temporal_split, only keep those retain users
    # whose post-processed basket list is at least length 4
    valid_retain = []
    for u in retain_user_ids:
        # baskets without the leading/trailing [-1]
        h_baskets = data_history[u][1:-1]
        # append their future basket so we get the full sequence
        h_baskets = h_baskets + [ data_future[u][1] ]
        # scif doesn't remove any items from h_baskets (retrain_flag=False)
        # check the minimal length requirement
        if len(h_baskets) >= 4:
            valid_retain.append(u)

    # overwrite retain_user_ids with only the valid ones
    retain_user_ids = valid_retain

    for i, user in enumerate(sorted(unlearning_user_ids)):
        epoch_start = time.perf_counter()
        clean_user_ids = []
        # check if cleaned history can be used for training:
        h_baskets = data_history[user][1:-1]
        # append their future basket so we get the full sequence
        h_baskets = h_baskets + [data_future[user][1]]
        # we chose a user which was filtered out
        if len(h_baskets) < 4:
            continue
        h_baskets = [[item for item in basket if item not in user_to_unlearning_items[user]] for basket in h_baskets]
        h_baskets = [basket for basket in h_baskets if len(basket) > 0]
        # scif doesn't remove any items from h_baskets (retrain_flag=False)
        # check the minimal length requirement
        if len(h_baskets) >= 4:
            clean_user_ids.append(user)

        print(f"\nunlearning items for user {i + 1}/{len(unlearning_user_ids)} with id: {user}\n")
        cur_unlearning_user_ids = [user]
        assert temporal_split, f"user split not implemented yet, need to set temporal_split."

        if args.unlearning_algorithm == "scif":
            scif.scif_unlearn(
                unlearning_user_ids=cur_unlearning_user_ids,
                retain_user_ids=retain_user_ids,
                n=len(user_list),
                model=model,
                criterion=loss_func,
                LOCAL=LOCAL,
                temporal_split=temporal_split,
                damping=0.01,
                scale=25.0,
                lissa_bs=16,
                retain_samples_used_for_update=args.retain_samples_used_for_update,
                train_pair_count=args.lissa_train_pair_count_scif,
                history_path=history_path,
                future_path=future_path,
                user_to_unlearning_items=user_to_unlearning_items,
            )
        elif args.unlearning_algorithm == "fanchuan":
            fanchuan.unlearn_neurips_iterative_contrastive(
                model=model,
                optimizer=optimizer,
                device=torch.device("cuda" if get_attribute("cuda") != -1 and torch.cuda.is_available() else "cpu"),
                LOCAL=LOCAL,
                history_path=history_path,
                future_path=future_path,
                unlearning_user_ids=cur_unlearning_user_ids,
                retain_user_ids=retain_user_ids,
                user_to_unlearning_items=user_to_unlearning_items,
                trainable_cleaned_unlearn_user_ids=clean_user_ids,
            )
        elif args.unlearning_algorithm == "kookmin":
            kookmin.unlearn_by_reinit_and_finetune(
                unlearning_user_ids=cur_unlearning_user_ids,
                retain_user_ids=retain_user_ids,
                model=model,
                criterion=loss_func,
                LOCAL=LOCAL,
                temporal_split=temporal_split,
                kookmin_init_rate=args.kookmin_init_rate,     # 1 % of lowest-|grad| params
                device=torch.device("cuda" if get_attribute("cuda") != -1 and torch.cuda.is_available() else "cpu"),
                optimizer=optimizer,
                param_list=param_list,
                param_index=param_index,
                retain_samples_used_for_update=args.retain_samples_used_for_update,
                history_path=history_path,
                future_path=future_path,
                user_to_unlearning_items=user_to_unlearning_items,
                trainable_cleaned_unlearn_user_ids=clean_user_ids,
            )

        epoch_elapsed = time.perf_counter() - epoch_start
        print(f"Epoch {i} took {epoch_elapsed:.2f} s")

        if i in checkpoint_idxs:
            unlearn_str = (
                f"_sensitive_category_{args.sensitive_category}"
                f"_unlearning_fraction_{args.unlearning_fraction}"
                f"_unlearning_algorithm_{args.unlearning_algorithm}"
            )
            model_path = f'{model_folder}/unlearn_model_best_epoch_{i}_seed_{args.seed}{unlearn_str}.pkl'
            save_model(model, model_path)
            print(f"saved model at checkpoint with idx: {i}")
        
        sys.stdout.flush()
    
    total_elapsed = time.perf_counter() - script_start
    print(f"All done in {total_elapsed:.2f} s")

    print("Evaluating the test set")

    for i in checkpoint_idxs:
        unlearn_str = (
            f"_sensitive_category_{args.sensitive_category}"
            f"_unlearning_fraction_{args.unlearning_fraction}"
            f"_unlearning_algorithm_{args.unlearning_algorithm}"
        )
        model_path = f'{model_folder}/unlearn_model_best_epoch_{i}_seed_{args.seed}{unlearn_str}.pkl'
        scores = evaluate_best_model(
            model=model,
            args=args,
            users_in_unlearning_set=unlearning_user_ids,
            user_to_unlearning_items=user_to_unlearning_items,
            model_path=model_path,
            retrain_str="",
            model_folder=model_folder,
            temporal_split=temporal_split,
            retrain_flag=True,
        )
        print("\n\n")


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    unlearn(save_model_folder=args.save_model_folder,
          history_path=args.history_path,
          future_path=args.future_path,
          keyset_path=args.keyset_path,
          item_embed_dim=args.item_embed_dim,
          loss_function=args.loss_function,
          epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.learning_rate,
          optim=args.optim,
          weight_decay=args.weight_decay,
          data_path=args.data_path,
          LOCAL=args.LOCAL,
          temporal_split=args.temporal_split,
          args=args)
    sys.exit()
