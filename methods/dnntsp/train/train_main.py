import sys
import pickle
import os
import torch

sys.path.append("..")
DEBUG = False
if DEBUG:
    sys.path.append("./methods/dnntsp")
    sys.path.append("./methods/dnntsp/train")

from train_model import train_model
from model.temporal_set_prediction import temporal_set_prediction
from utils.util import get_class_weights
from utils.util import convert_to_gpu, convert_all_data_to_gpu
from utils.metric import get_metric
from utils.loss import BPRLoss, WeightMSELoss
from utils.data_container import get_data_loader, get_data_loader_temporal_split
from utils.load_config import get_attribute
import torch
import torch.nn as nn
import os
import shutil
import argparse
import math


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

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
        "--retrain_flag",
        action="store_true",
        help="set this flag if you want to retrain the model without a certain forget set",
    )
    parser.add_argument(
        "--retrain_checkpoint_idx_to_match",
        type=int,
        default=None,
        help="which unlearning checkpoint should be taken as example to create the unlearning set (only take a subset of the unlearning set given)"
    )
    parser.add_argument(
        "--sensitive_category",
        type=str,
        default=None,
        choices=["baby", "meat", "alcohol"],
        help="When choosing sensitive items to unlearn, choose which category"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="popular",
        help="Unlearning method to use: random, popular, unpopular, or sensitive"
    )
    parser.add_argument(
        "--unlearning_fraction",
        type=float,
        default=0.0001,
        help="Fraction of data to unlearn"
    )
    parser.add_argument(
        "--popular_percentage",
        type=float,
        default=0.1,
        help="Fraction of most/least popular items to consider"
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


def evaluate_best_model(model,
                        args,
                        users_in_unlearning_set=None,
                        user_to_unlearning_items=None,
                        retrain_str="",
                        model_folder=None,
                        temporal_split=False,
                        model_path=None,
                        retrain_flag=False,
                        seed=None,
                        history_path=None,
                        future_path=None,
                        keyset_path=None,
                        batch_size=None,):
    """
    Load the best checkpoint saved during training and evaluate it on the TEST split,
    reporting Recall@10, Recall@20, NDCG@10, NDCG@20, HitRate@10 and HitRate@20.
    
    Parameters
    ----------
    model : nn.Module
        The uninitialized model instance (same class you trained).
    args : argparse.Namespace
        The parsed CLI arguments from parse_args().
    users_in_unlearning_set : list or None
        If you used retrain_flag, pass through your computed list.
    user_to_unlearning_items : dict or None
        If you used retrain_flag, pass through your mapping.
    retrain_str : str
        Your retrain_str suffix (empty if none).
    model_folder : str
        The folder where your checkpoints were saved 
        (must match the one you passed into train_model).
    
    Returns
    -------
    dict
        A mapping from metric name (e.g. 'recall_10') to its value.
    """

    seed = seed or args.seed
    history_path = history_path or args.history_path
    future_path = future_path or args.future_path
    batch_size = batch_size or args.batch_size
    #keyset_path = keyset_path or args.keyset_path

    # 1) build test DataLoader
    if temporal_split:
        test_loader = get_data_loader_temporal_split(
            history_path=history_path,
            future_path=future_path,
            data_type='test',
            batch_size=batch_size,
            item_embedding_matrix=model.item_embedding,
            retrain_flag=retrain_flag,
            users_in_unlearning_set=users_in_unlearning_set,
            user_to_unlearning_items=user_to_unlearning_items,
        )
    else:
        test_loader = get_data_loader(
            history_path=history_path,
            future_path=future_path,
            keyset_path=keyset_path,
            data_type='test',
            batch_size=batch_size,
            item_embedding_matrix=model.item_embedding
        )

    # 2) load best checkpoint
    best_path = os.path.join(
        model_folder,
        f"model_best_seed_{seed}{retrain_str}.pkl"
    ) if model_path is None else model_path
    print(f"[evaluate] Loading model from: {best_path}")
    # re‐construct model architecture
    model.load_state_dict(torch.load(best_path))
    model = convert_to_gpu(model)
    model.eval()

    # 3) inference
    all_y_true, all_y_pred = [], []
    with torch.no_grad():
        for (g, nodes_feature, edges_weight,
             lengths, nodes, truth_data, users_frequency) in test_loader:

            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(
                    g, nodes_feature, edges_weight,
                    lengths, nodes, truth_data, users_frequency
                )

            out = model(
                g, nodes_feature, edges_weight,
                lengths, nodes, users_frequency
            )
            all_y_pred.append(out.cpu())
            all_y_true.append(truth_data.cpu())

    y_true = torch.cat(all_y_true, dim=0)
    y_pred = torch.cat(all_y_pred, dim=0)

    # 4) compute metrics
    scores = get_metric(y_true=y_true, y_pred=y_pred)
    results = {}
    for k in [10, 20]:
        r = scores.get(f"recall_{k}")
        n = scores.get(f"ndcg_{k}")
        h = scores.get(f"PHR_{k}")
        print(f"--- Test @ {k} ---")
        if r is not None: print(f"Recall@{k}:  {r:.4f}")
        if n is not None: print(f"NDCG@{k}:    {n:.4f}")
        if h is not None: print(f"HitRate@{k}: {h:.4f}")
        if r is not None: results[f"recall_{k}"] = r
        if n is not None: results[f"ndcg_{k}"]    = n
        if h is not None: results[f"hitrate_{k}"] = h

    return results


def train(
    save_model_folder,
    history_path,
    future_path,
    keyset_path,
    item_embed_dim,
    loss_function,
    epochs,
    batch_size,
    learning_rate,
    optim,
    weight_decay,
    data_path,
    LOCAL=False,
    temporal_split=False,
    retrain_flag=False,
    retrain_checkpoint_idx_to_match=None,
    sensitive_category=None,
    method=None,
    popular_percentage=None,
    unlearning_fraction=None,
    seed=None,
    args=None,
):
    model = create_model(save_model_folder, item_embed_dim)

    dataset = get_attribute("data")
    percentage_str = f"_popular_percentage_{popular_percentage}" if method in ["popular", "unpopular"] else ""
    fraction_str = f"_unlearning_fraction_{unlearning_fraction}" if method in ["popular", "unpopular", "random", "sensitive"] else ""
    unlearning_data_file = f'../../../unlearning_data/dataset_{dataset.lower()}_seed_{seed}_method_{method}{fraction_str}{percentage_str}.pkl'

    users_in_unlearning_set = None
    user_to_unlearning_items = None
    retrain_str = ""

    if retrain_flag:

        with open(unlearning_data_file, "rb") as f:
            user_to_unlearning_items = pickle.load(f)
            if method == "sensitive":
                user_to_unlearning_items = user_to_unlearning_items[sensitive_category]
        
        retrain_str = (
            f"_sensitive_category_{sensitive_category}"
            f"_unlearning_fraction_{unlearning_fraction}"
            f"_retrain_checkpoint_idx_to_match_{retrain_checkpoint_idx_to_match}"
        )
        # remove current forget set from the training data. need parameter to tell how much of the forget set is taken to get the wanted retrained models at certain subsets of the unlearning set
        n = len(user_to_unlearning_items)
        checkpoint_every = math.ceil(n / 4)
        checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]

        if len(checkpoint_idxs) == 5:
            checkpoint_idxs = checkpoint_idxs[:4] + [checkpoint_idxs[-1]]

        unlearning_set_take_first_x = checkpoint_idxs[retrain_checkpoint_idx_to_match]
        # remove sensitive items from users in retraining
        users_in_unlearning_set = sorted(user_to_unlearning_items.keys())[:unlearning_set_take_first_x + 1]

    if temporal_split:
        train_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=batch_size,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=retrain_flag,
                                            users_in_unlearning_set=users_in_unlearning_set,
                                            user_to_unlearning_items=user_to_unlearning_items,)
        valid_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='val',
                                            batch_size=batch_size,
                                            item_embedding_matrix=model.item_embedding,
                                            retrain_flag=retrain_flag,
                                            users_in_unlearning_set=users_in_unlearning_set,
                                            user_to_unlearning_items=user_to_unlearning_items,)
    else:
        train_data_loader = get_data_loader(history_path=history_path,
                                            future_path=future_path,
                                            keyset_path=keyset_path,
                                            data_type='train',
                                            batch_size=batch_size,
                                            item_embedding_matrix=model.item_embedding)
        valid_data_loader = get_data_loader(history_path=history_path,
                                            future_path=future_path,
                                            keyset_path=keyset_path,
                                            data_type='val',
                                            batch_size=batch_size,
                                            item_embedding_matrix=model.item_embedding)
        
    loss_func = create_loss(loss_function, data_path)

    data = get_attribute("data")
    
    if LOCAL:
        model_folder = f"../save_model_folder/{data}/{save_model_folder}"
        tensorboard_folder = f"../results/runs/{data}/{save_model_folder}"
    else:
        model_folder = f"/opt/results/save_model_folder/{data}/{save_model_folder}"
        tensorboard_folder = f"/opt/results/runs/{data}/{save_model_folder}"

    # shutil.rmtree(model_folder, ignore_errors=True)
    os.makedirs(model_folder, exist_ok=True)
    # shutil.rmtree(tensorboard_folder, ignore_errors=True)
    os.makedirs(tensorboard_folder, exist_ok=True)

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

    train_model(model=model,
                train_data_loader=train_data_loader,
                valid_data_loader=valid_data_loader,
                loss_func=loss_func,
                epochs=epochs,
                optimizer=optimizer,
                model_folder=model_folder,
                tensorboard_folder=tensorboard_folder,
                LOCAL=LOCAL,
                retrain_flag=retrain_flag,
                retrain_str=retrain_str,
                seed=seed,)
    
    scores = evaluate_best_model(
        model=model,
        args=args,
        users_in_unlearning_set=users_in_unlearning_set,
        user_to_unlearning_items=user_to_unlearning_items,
        retrain_str=retrain_str,
        model_folder=model_folder,
        temporal_split=temporal_split,
        retrain_flag=args.retrain_flag,
    )

    

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    train(save_model_folder=args.save_model_folder,
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
        retrain_flag=args.retrain_flag,
        seed=args.seed,
        retrain_checkpoint_idx_to_match=args.retrain_checkpoint_idx_to_match,
        sensitive_category=args.sensitive_category,
        method=args.method,
        popular_percentage=args.popular_percentage,
        unlearning_fraction=args.unlearning_fraction,
        args=args,
    )
    sys.exit()
