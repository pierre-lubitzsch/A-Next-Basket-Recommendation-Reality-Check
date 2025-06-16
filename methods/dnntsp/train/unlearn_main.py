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


def unlearn(save_model_folder, history_path, future_path, keyset_path, item_embed_dim, loss_function, epochs, batch_size, learning_rate, optim, weight_decay, data_path, LOCAL=False, temporal_split=False, retrain_flag=False, unlearn_flag=False):
    model = create_model(save_model_folder, item_embed_dim)

    # either retrain or unlearn
    assert unlearn_flag ^ retrain_flag
    
    if temporal_split:
        train_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='train',
                                            batch_size=batch_size,
                                            item_embedding_matrix=model.item_embedding)
        valid_data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='val',
                                            batch_size=batch_size,
                                            item_embedding_matrix=model.item_embedding)
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

    shutil.rmtree(model_folder, ignore_errors=True)
    os.makedirs(model_folder, exist_ok=True)
    shutil.rmtree(tensorboard_folder, ignore_errors=True)
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
                LOCAL=LOCAL)


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
