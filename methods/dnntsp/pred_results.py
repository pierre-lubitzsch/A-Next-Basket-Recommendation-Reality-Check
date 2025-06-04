import json
import sys
import os
from tqdm import tqdm
import torch

sys.path.append("./train")

from utils.metric import evaluate
from utils.data_container import get_data_loader, get_data_loader_temporal_split
from utils.load_config import get_attribute
from utils.util import convert_to_gpu
from train.train_main import create_model
from utils.util import load_model

import argparse

DEBUG = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunnhumby', help='Dataset')
    parser.add_argument('--fold_id', type=int, default=0, help='x')
    parser.add_argument('--best_model_path', type=str, required=True)
    parser.add_argument('--temporal_split', action="store_true", help="when set, use a temporal split for the dataset. otherwise use a user split")
    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold_id
    model_path = args.best_model_path
    seed = int(model_path.split("/")[-2].split("seed_")[-1])

    if DEBUG:
        history_path = f'./jsondata/{dataset}_history.json'
        future_path = f'./jsondata/{dataset}_future.json'
        keyset_path = f'./keyset/{dataset}_keyset_{fold}.json'
        model_path = "./methods/dnntsp/" + model_path
    else:
        history_path = f'../../jsondata/{dataset}_history.json'
        future_path = f'../../jsondata/{dataset}_future.json'
        keyset_path = f'../../keyset/{dataset}_keyset_{fold}.json'
    
    pred_path = f'{dataset}_pred{fold}_100_epochs_seed_{seed}.json'
    truth_path = f'{dataset}_truth{fold}_100_epochs_seed_{seed}.json'
    
    with open(keyset_path, 'r') as f:
        keyset = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(save_model_folder="./save_model_folder", item_embed_dim=get_attribute("item_embed_dim"))
    model = load_model(model, model_path)
    
    

    if args.temporal_split:
        data_loader = get_data_loader_temporal_split(history_path=history_path,
                                            future_path=future_path,
                                            data_type='test',
                                            batch_size=1,
                                            item_embedding_matrix=model.item_embedding)
    else:
        data_loader = get_data_loader(history_path=history_path,
                                        future_path=future_path,
                                    keyset_path=keyset_path,
                                    data_type='test',
                                    batch_size=1,
                                    item_embedding_matrix=model.item_embedding)
    

    if get_attribute("cuda") != -1:
        model = model.to(device)

    model.eval()

    pred_dict = dict()
    truth_dict = dict()
    if args.temporal_split:
        with open(future_path, 'r') as f:
            future = json.load(f)
        test_key = sorted(future.keys(), key=int)
    else:
        test_key = keyset['test']
    user_ind = 0
    for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                    tqdm(data_loader)):
        if get_attribute("cuda") != -1:
            g = g.to(device)
            nodes_feature = nodes_feature.to(device)
            edges_weight = edges_weight.to(device)
            lengths = lengths.to(device)
            nodes = nodes.to(device)
            truth_data = truth_data.to(device)
            users_frequency = users_frequency.to(device)
        
        
        pred_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
        pred_list = pred_data.detach().squeeze(0).cpu().numpy().argsort()[::-1][:100].tolist()
        truth_list = truth_data.detach().squeeze(0).cpu().numpy().argsort()[::-1][:100].tolist()
        pred_dict[test_key[user_ind]] = pred_list
        truth_dict[test_key[user_ind]] = truth_list
        user_ind += 1

        if get_attribute("cuda") != -1:
            g = g.cpu()
            nodes_feature = nodes_feature.cpu()
            edges_weight = edges_weight.cpu()
            lengths = lengths.cpu()
            nodes = nodes.cpu()
            truth_data = truth_data.cpu()
            users_frequency = users_frequency.cpu()

    with open(pred_path, 'w') as f:
        json.dump(pred_dict, f)
    with open(truth_path, 'w') as f:
        json.dump(truth_dict, f)

