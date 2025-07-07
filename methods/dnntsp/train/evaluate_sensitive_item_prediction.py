#!/usr/bin/env python3

import argparse, os, re, pickle, glob, sys
from pathlib import Path
import pandas as pd, torch, tqdm

from train_main import evaluate_best_model

sys.path.append("..")

from utils.load_config import get_attribute
from utils.util import convert_to_gpu, convert_all_data_to_gpu, load_model
from model.temporal_set_prediction import temporal_set_prediction
from utils.data_container import get_data_loader_temporal_split

# ----------------------------------------------------------------------
# discovery helpers
# ----------------------------------------------------------------------
CKPT_RGX = (r"unlearn_model_best_epoch_(\d+)_seed_(\d+)"
            r"_sensitive_category_([\w\-]+)"
            r"_unlearning_fraction_([\d\.]+)"
            r"_unlearning_algorithm_([\w\-]+)\.pkl")

RETRAIN_RGX = (r"model_best_seed_(\d+)"
            r"_sensitive_category_([\w\-]+)"
            r"_unlearning_fraction_([\d\.]+)"
            r"_retrain_checkpoint_idx_to_match_(\d+)\.pkl")

VALID_BASELINE_SEEDS = {2, 3, 5, 7, 11}

def list_run_dirs(root: Path):
    for p in [root] + [d for d in root.iterdir() if d.is_dir()]:
        if any(
            re.match(CKPT_RGX, f.name) or re.match(r"model_best_seed_(\d+)\.pkl", f.name)
            for f in p.glob("*.pkl")
        ):
            yield p

def discover_ckpts(run_dir: Path):
    ckpt_paths = sorted(glob.glob(str(run_dir / "*.pkl")))
    return [
        p for p in ckpt_paths
        if re.search(CKPT_RGX, os.path.basename(p))
            or re.search(RETRAIN_RGX, os.path.basename(p))
            or re.match(r"model_best_seed_(\d+)\.pkl", os.path.basename(p))
    ]

def parse_ckpt(fname):
    base = os.path.basename(fname)
    m = re.search(CKPT_RGX, base)
    if m:
        return [dict(epoch=int(m.group(1)), seed=int(m.group(2)),
                     category=m.group(3), unlearning_fraction=float(m.group(4)),
                     algorithm=m.group(5))]
    
    m2 = re.match(r"model_best_seed_(\d+)\.pkl", base)
    if m2:
        seed = int(m2.group(1))
        if seed in VALID_BASELINE_SEEDS:
            return [
                dict(epoch=-1, seed=seed, category=cat,
                     unlearning_fraction=0.001, algorithm="baseline")
                for cat in ["baby", "meat", "alcohol"]
            ]
        

    m3 = re.search(RETRAIN_RGX, base)
    if m3:
        seed = int(m3.group(1))
        category = m3.group(2)
        unlearning_fraction = float(m3.group(3))

        unlearning_data_file = (f"../../../unlearning_data/dataset_"
                                f"instacart_seed_{seed}"
                                f"_method_sensitive_unlearning_fraction_{unlearning_fraction}.pkl")
        with open(unlearning_data_file, "rb") as f:
            user_to_unlearning_items = pickle.load(f)
            user_to_unlearning_items = user_to_unlearning_items[category]

        n = len(user_to_unlearning_items)
        checkpoint_every = (n + 3) // 4 # ceil
        checkpoint_idxs = [i for i in range(n) if i > 0 and ((i <= 3 * n // 4 + 5 and i % checkpoint_every == 0) or (i >= 3 * n // 4 + 5 and i == n - 1))]
        idx = int(m3.group(4))
        epoch = checkpoint_idxs[idx]
        return [dict(epoch=epoch, seed=seed,
                     category=category, unlearning_fraction=unlearning_fraction,
                     algorithm="retrain")]

    return []

# ----------------------------------------------------------------------
# model helpers
# ----------------------------------------------------------------------
def build_model(embed_dim: int):
    return convert_to_gpu(temporal_set_prediction(
        items_total=get_attribute("items_total"),
        item_embedding_dim=embed_dim))

def load_sensitive_items(ds, seed, frac, cat):
    fn = (f"../../../unlearning_data/dataset_{ds.lower()}_seed_{seed}"
          f"_method_sensitive_unlearning_fraction_{frac}.pkl")
    with open(fn, "rb") as f:
        mapping = pickle.load(f)[cat]
    return set(i for v in mapping.values() for i in v), mapping

# ----------------------------------------------------------------------
# core counting logic
# ----------------------------------------------------------------------
def count_sensitive_users(model, loader, sensitive_items, k):
    """
    Return (#users with ≥1 sensitive item in top-k, total #users).
    """
    model.eval()
    n_flagged = n_total = 0
    sens_set = sensitive_items  # local alias (Python set lookup is fast)

    with torch.no_grad():
        for (g, nf, ew, L, n, y, uf) in loader:
            # send to GPU
            g, nf, ew, L, n, y, uf = convert_all_data_to_gpu(
                g, nf, ew, L, n, y, uf)

            # forward
            logits = model(g, nf, ew, L, n, uf)
            topk = torch.topk(logits, k, dim=1).indices.cpu()  # shape [B, k]

            # per-row membership test
            for row in topk:
                if any(idx.item() in sens_set for idx in row):
                    n_flagged += 1
            n_total += topk.size(0)

            # release GPU memory early
            del logits, g, nf, ew, L, n, y, uf, topk
            torch.cuda.empty_cache()

    return n_flagged, n_total

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../save_model_folder/Instacart/")
    ap.add_argument("--history_path", default="../../../jsondata/instacart_history.json")
    ap.add_argument("--future_path", default="../../../jsondata/instacart_future.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--item_embed_dim", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--performance_evaluation", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ds_name = get_attribute("data")
    rows = []
    original_models = [f"model_best_seed_{s}.pkl" for s in [2, 3, 5, 7, 11]]

    category_to_items = {
        "meat": [5, 95, 96, 15, 33, 34, 35, 49, 106, 122],
        "alcohol": [27, 28, 62, 124, 134],
        "baby": [82, 92, 102, 56],
    }

    for run_dir in list_run_dirs(root):
        for ckpt in discover_ckpts(run_dir):
            metas = parse_ckpt(ckpt)
            for meta in metas:
                try:
                    model = build_model(args.item_embed_dim)
                    model = load_model(model, ckpt)

                    sens_set, user2items = load_sensitive_items(
                        ds_name, meta["seed"], meta["unlearning_fraction"], meta["category"])

                    # filter out users which were not unlearned yet
                    filename = ckpt.split("/")[-1]
                    unlearn_first_x = len(user2items) if meta["epoch"] == -1 else meta["epoch"] + 1
                    users = set(sorted(user2items.keys())[:unlearn_first_x])
                    user2items = {k: v for k, v in user2items.items() if k in users}

                    if args.performance_evaluation and any([x in filename or not x.startswith("unlearn") for x in original_models]):
                        seed = meta["seed"]
                        scores = evaluate_best_model(
                            model=model,
                            args=args,
                            users_in_unlearning_set=list(user2items.keys()),
                            user_to_unlearning_items=user2items,
                            retrain_str="",
                            model_path=ckpt,
                            temporal_split=True,
                            retrain_flag=True,
                            seed=seed,
                            history_path=args.history_path,
                            future_path=args.future_path,
                            batch_size=args.batch_size,
                        )

                    loader = get_data_loader_temporal_split(
                        history_path=args.history_path,
                        future_path=args.future_path,
                        data_type="test",
                        batch_size=args.batch_size,
                        item_embedding_matrix=model.item_embedding,
                        retrain_flag=True,
                        users_in_unlearning_set=list(user2items.keys()),
                        user_to_unlearning_items=user2items,
                        user_subset=list(user2items.keys()))

                    tqdm_loader = tqdm.tqdm(loader, leave=False, disable=True)

                    n_flagged, n_users = count_sensitive_users(model, tqdm_loader, category_to_items[meta["category"]], args.top_k)

                    print(f"[{run_dir.name} | {os.path.basename(ckpt)} | filtered data | cat={meta['category']}]\n"
                          f"{n_flagged}/{n_users} users flagged (top-{args.top_k})")

                    rows.append({
                        **meta,
                        "run_dir": run_dir.name,
                        "ckpt_file": os.path.basename(ckpt),
                        "users_with_sensitive_predictions": n_flagged,
                        "total_users": n_users,
                        "sensitive_items_removed": True,
                    })

                    loader = get_data_loader_temporal_split(
                        history_path=args.history_path,
                        future_path=args.future_path,
                        data_type="test",
                        batch_size=args.batch_size,
                        item_embedding_matrix=model.item_embedding,
                        retrain_flag=False,
                        users_in_unlearning_set=list(user2items.keys()),
                        user_to_unlearning_items=user2items,
                        user_subset=list(user2items.keys()))

                    tqdm_loader = tqdm.tqdm(loader, leave=False, disable=True)

                    n_flagged, n_users = count_sensitive_users(model, tqdm_loader, category_to_items[meta["category"]], args.top_k)

                    print(f"[{run_dir.name} | {os.path.basename(ckpt)} | original data | cat={meta['category']}]\n"
                          f"{n_flagged}/{n_users} users flagged (top-{args.top_k})")

                    rows.append({
                        **meta,
                        "run_dir": run_dir.name,
                        "ckpt_file": os.path.basename(ckpt),
                        "users_with_sensitive_predictions": n_flagged,
                        "total_users": n_users,
                        "sensitive_items_removed": False,
                    })


                except Exception as e:
                    print(f"Skipped {ckpt} ({meta['category']}) due to error: {e}")

                finally:
                    del model, loader
                    torch.cuda.empty_cache()
                    sys.stdout.flush()

    out_csv = root / "sensitive_predictions_summary_v3.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSummary saved to {out_csv}")

if __name__ == "__main__":
    main()
