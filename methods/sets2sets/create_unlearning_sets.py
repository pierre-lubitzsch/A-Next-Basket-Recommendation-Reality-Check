import json
import torch
import numpy as np
import argparse
import collections
import random
import pickle
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Parse unlearning parameters")
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
        "--seed",
        type=int,
        default=2,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tafeng",
        help="Dataset name"
    )
    parser.add_argument(
        "--popular_percentage",
        type=float,
        default=0.1,
        help="Fraction of most/least popular items to consider"
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=1000,
        help="How many users to sample for unlearning set selection"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    history_file = f"../../jsondata/{args.dataset}_history.json"
    future_file = f"../../jsondata/{args.dataset}_future.json"          



    with open(history_file, 'r') as f:
        history_data = json.load(f)
    with open(future_file, 'r') as f:
        future_data = json.load(f)

    user_list = [x for x in future_data if len(history_data[x]) - 2 + len(future_data[x]) - 2 >= 4]

    item_count = collections.defaultdict(int)
    user_item_count = collections.defaultdict(int)

    for user in user_list:
        for basket in history_data[user][1:-2]:
            user_item_count[user] += len(basket)
            for item in basket:
                item_count[item] += 1

    total_item_count = sum(item_count.values())
    wanted_unlearning_count = int(total_item_count * args.unlearning_fraction)
    user_to_unlearn_items = collections.defaultdict(list)
    if args.method in ["random", "popular", "unpopular"]:

        if args.method == "random":
            candidate_users = user_list
            user_weights = [user_item_count[user] / total_item_count for user in candidate_users]
            item_filter = None

        else:
            sorted_items = sorted(item_count.items(), key=lambda x: x[1], reverse=(args.method == "popular"))
            top_k = max(1, int(len(sorted_items) * args.popular_percentage))
            selected_items = set(item for item, _ in sorted_items[:top_k])
            item_filter = selected_items

            candidate_users = []
            for user in user_list:
                user_items = {item for basket in history_data[user][1:-2] for item in basket}
                if user_items & selected_items:
                    candidate_users.append(user)

            if not candidate_users:
                print(f"No users found with {args.method} items. Try increasing --popular_percentage.")
                return
            user_weights = [1.0 for _ in candidate_users]

        user_to_max_count = {}
        for user in candidate_users:
            unique_items = set(item for basket in history_data[user][1:-2] for item in basket if (item_filter is None or item in item_filter))
            user_to_max_count[user] = max(1, len(unique_items))

        user_to_current_count = collections.defaultdict(int)
        total_unlearned_items = 0

        while total_unlearned_items < wanted_unlearning_count:
            user = random.choices(candidate_users, weights=user_weights, k=1)[0]
            if user_to_current_count[user] >= user_to_max_count[user]:
                continue

            cur_item_to_count = collections.defaultdict(int)
            baskets = history_data[user][1:-2]
            for basket in baskets:
                for item in basket:
                    if item_filter is None or item in item_filter:
                        cur_item_to_count[item] += 1

            if not cur_item_to_count:
                continue

            cur_items = list(cur_item_to_count.keys())
            cur_weights = [cur_item_to_count[item] for item in cur_items]
            item = random.choices(cur_items, weights=cur_weights, k=1)[0]

            total_unlearned_items += cur_item_to_count[item]
            user_to_current_count[user] += 1
            user_to_unlearn_items[user].append(item)

            if all(user_to_current_count[u] >= user_to_max_count[u] for u in candidate_users):
                print("Warning: All users hit cap before reaching total unlearning target.")
                break
    # elif args.method == "sensitive":
        

    #     dataset_to_sensitive_aisles = {
    #         "instacart": {
    #             "meat": [5, 95, 96, 96, 15, 33, 34, 35, 49, 106, 122],
    #             "alcohol": [27, 28, 62, 124, 134],
    #             "baby": [82, 92, 102, 56],
    #         }
    #     }

    #     products_with_aisle_id_filepath = f"../../dataset/{args.dataset}_products.csv"
    #     products = pd.read_csv(products_with_aisle_id_filepath)

    #     sensitive_categories_to_product_ids = dict()

    #     for sensitive_category in dataset_to_sensitive_aisles[args.dataset].keys():
    #         sensitive_products = products[products["aisle_id"].isin(dataset_to_sensitive_aisles[args.dataset][sensitive_category])]
    #         sensitive_categories_to_product_ids[sensitive_category] = list(sensitive_products["product_id"])

    #     user_to_unlearn_items = dict()
    #     for sensitive_category in dataset_to_sensitive_aisles[args.dataset].keys():
    #         sensitive_product_ids_set = set(sensitive_categories_to_product_ids[sensitive_category])
    #         user_to_unlearn_items[sensitive_category] = dict()
    #         for user in user_list:
    #             user_to_unlearn_items[sensitive_category][user] = set()
    #             for basket in history_data[user][1:-2]:
    #                 user_to_unlearn_items[sensitive_category][user] |= set(basket) & sensitive_product_ids_set
    #             if len(user_to_unlearn_items[sensitive_category][user]) == 0:
    #                 del user_to_unlearn_items[sensitive_category][user]
    #             else:
    #                 user_to_unlearn_items[sensitive_category][user] = sorted(user_to_unlearn_items[sensitive_category][user])

    elif args.method == "sensitive":

        # Instacart sensitive items: aisle_ids
        # 5,marinades meat preparation
        # 95,canned meat seafood
        # 96,lunch meat
        # 15,packaged seafood
        # 33,kosher foods
        # 34,frozen meat seafood
        # 35,poultry counter
        # 49,packaged poultry
        # 106,hot dogs bacon sausage
        # 122,meat counter

        # 27,beers coolers
        # 28,red wines
        # 62,white wines
        # 124,spirits
        # 134,specialty wines champagnes

        dataset_to_sensitive_aisles = {
            "instacart": {
                "meat": [5, 95, 96, 15, 33, 34, 35, 49, 106, 122],
                "alcohol": [27, 28, 62, 124, 134],
                "baby": [82, 92, 102, 56],
            }
        }

        products_with_aisle_id_filepath = f"../../dataset/{args.dataset}_products.csv"
        products = pd.read_csv(products_with_aisle_id_filepath)

        sensitive_categories_to_product_ids = {
            cat: set(products[products["aisle_id"].isin(aisle_ids)]["product_id"])
            for cat, aisle_ids in dataset_to_sensitive_aisles[args.dataset].items()
        }

        # 1. Compute total number of items in all history baskets
        total_item_count = sum(len(basket) for user in user_list for basket in history_data[user][1:-2])
        wanted_unlearning_count = int(args.unlearning_fraction * total_item_count)
        print(f"Total item count: {total_item_count}, Unlearning target: {wanted_unlearning_count}")

        user_to_unlearn_items = dict()

        for category, sensitive_product_ids in sensitive_categories_to_product_ids.items():
            eligible_users = dict()
            for user in user_list:
                user_sensitive_items = set()
                for basket in history_data[user][1:-2]:
                    user_sensitive_items |= set(basket) & sensitive_product_ids
                if user_sensitive_items:
                    eligible_users[user] = sorted(user_sensitive_items)


            print(f"[{category}] Eligible users: {len(eligible_users)}")

            # 2. Sample users until unlearning threshold is met (no trimming)
            selected_users = dict()
            total_unlearned = 0
            user_pool = list(eligible_users.keys())
            random.shuffle(user_pool)

            for user in user_pool:
                items = eligible_users[user]
                selected_users[user] = items
                total_unlearned += len(items)
                if total_unlearned >= wanted_unlearning_count:
                    break

            print(f"[{category}] Selected {len(selected_users)} users, total sensitive items: {total_unlearned}")
            user_to_unlearn_items[category] = selected_users
   
            
    else:
        print(f"Unknown method: {args.method}")
        return

    print(f"Total users: {len(user_list)}")
    print(f"Unlearning target: {wanted_unlearning_count} items")

    save_dir = "../../unlearning_data/"
    os.makedirs(save_dir, exist_ok=True)
    percentage_str = f"_popular_percentage_{args.popular_percentage}" if args.method in ["popular", "unpopular"] else ""
    fraction_str = f"_unlearning_fraction_{args.unlearning_fraction}" if args.method in ["popular", "unpopular", "random", "sensitive"] else ""

    out_file = f"dataset_{args.dataset}_seed_{args.seed}_method_{args.method}{fraction_str}{percentage_str}.pkl"

    with open(os.path.join(save_dir, out_file), "wb") as f:
        pickle.dump(user_to_unlearn_items, f)

if __name__ == "__main__":
    main()
