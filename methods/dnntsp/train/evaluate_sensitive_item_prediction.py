#!/usr/bin/env python3
"""
Walk through every un-learning run below --root and count how many
sensitive items still appear in the model’s top-k predictions.
"""

import argparse
import os
import re
import pickle
import glob
import sys
from pathlib import Path

import pandas as pd
import torch
import tqdm

sys.path.append("..")                       # project’s root

from utils.load_config import get_attribute
from utils.util import convert_to_gpu, convert_all_data_to_gpu, load_model
from model.temporal_set_prediction import temporal_set_prediction
from utils.data_container import get_data_loader_temporal_split

# ----------------------------------------------------------------------
# discovery helpers
# ----------------------------------------------------------------------
CKPT_RGX = (
    r"unlearn_model_best_epoch_(\d+)_seed_(\d+)"
    r"_sensitive_category_([\w\-]+)"
    r"_unlearning_fraction_([\d\.]+)"
    r"_unlearning_algorithm_([\w\-]+)\.pkl"
)


def list_run_dirs(root: Path):
    """
    Yield every sub-directory under *root* that contains at least one
    checkpoint matching our naming convention.  If *root* itself already
    contains checkpoints, it is yielded as well.
    """
    for path in [root] + [p for p in root.iterdir() if p.is_dir()]:
        if any(re.match(CKPT_RGX, f.name) for f in path.glob("*.pkl")):
            yield path


def discover_model_files(run_dir: Path):
    """All unlearning checkpoints inside *run_dir* (non-recursive)."""
    pattern = run_dir / "unlearn_model_best_epoch_*_seed_*_sensitive_category_*_unlearning_fraction_*_unlearning_algorithm_*.pkl"
    return sorted(glob.glob(str(pattern)))


def parse_filename(fname: str):
    m = re.search(CKPT_RGX, os.path.basename(fname))
    if not m:
        raise ValueError(f"Cannot parse checkpoint name: {fname}")
    return dict(
        epoch=int(m.group(1)),
        seed=int(m.group(2)),
        category=m.group(3),
        unlearning_fraction=float(m.group(4)),
        algorithm=m.group(5),
    )


# ----------------------------------------------------------------------
# model + dataset helpers
# ----------------------------------------------------------------------
def build_model(item_embed_dim: int):
    items_total = get_attribute("items_total")
    model = temporal_set_prediction(
        items_total=items_total, item_embedding_dim=item_embed_dim
    )
    return convert_to_gpu(model)


def load_sensitive_items(dataset: str, seed: int, unlearning_fraction: float, category: str):
    pkl = (
        f"../../../unlearning_data/"
        f"dataset_{dataset.lower()}_seed_{seed}_method_sensitive"
        f"_unlearning_fraction_{unlearning_fraction}.pkl"
    )
    with open(pkl, "rb") as f:
        user_to_items = pickle.load(f)[category]

    sensitive = {item for items in user_to_items.values() for item in items}
    return sensitive, user_to_items


def count_sensitive_preds(model, loader, sensitive_items, k: int):
    model.eval()
    sens, tot = 0, 0

    with torch.no_grad():
        for (
            g,
            nodes_feature,
            edges_weight,
            lengths,
            nodes,
            truth_data,
            users_frequency,
        ) in loader:

            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = convert_all_data_to_gpu(
                g,
                nodes_feature,
                edges_weight,
                lengths,
                nodes,
                truth_data,
                users_frequency,
            )

            logits = model(
                g, nodes_feature, edges_weight, lengths, nodes, users_frequency
            )

            topk = torch.topk(logits, k, dim=1).indices.cpu().tolist()
            sens += sum(item in sensitive_items for row in topk for item in row)
            tot += k * len(topk)

    return sens, tot


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset folder (e.g. .../Instacart)")
    ap.add_argument("--history_path", required=True)
    ap.add_argument("--future_path", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--item_embed_dim", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    dataset_name = get_attribute("data")
    summary_rows = []

    # --------------------------------------------------
    # iterate over every run directory
    # --------------------------------------------------
    for run_dir in list_run_dirs(root):
        for ckpt in discover_model_files(run_dir):
            meta = parse_filename(ckpt)

            # 1) model
            model = build_model(args.item_embed_dim)
            model = load_model(model, ckpt)

            # 2) sensitive item universe
            sens_items, user_to_unlearning_items = load_sensitive_items(
                dataset_name,
                meta["seed"],
                meta["unlearning_fraction"],
                meta["category"],
            )

            # 3) test loader
            test_loader = get_data_loader_temporal_split(
                history_path=args.history_path,
                future_path=args.future_path,
                data_type="test",
                batch_size=args.batch_size,
                item_embedding_matrix=model.item_embedding,
                retrain_flag=True,
                users_in_unlearning_set=list(user_to_unlearning_items.keys()),
                user_to_unlearning_items=user_to_unlearning_items,
            )

            # 4) count
            s, t = count_sensitive_preds(model, tqdm.tqdm(test_loader, leave=False, disable=True), sens_items, args.top_k)
            print(f"[{run_dir.name} | {os.path.basename(ckpt)}]  {s}/{t} sensitive (top-{args.top_k})")

            summary_rows.append({**meta, "run_dir": run_dir.name, "sensitive_predictions": s, "total_predictions": t})

            # clean up GPU memory before the next run
            del model
            del test_loader
            torch.cuda.empty_cache()
            sys.stdout.flush()

    # --------------------------------------------------
    # write CSV
    # --------------------------------------------------
    out_csv = root / "sensitive_predictions_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nSummary written to {out_csv}")


if __name__ == "__main__":
    main()
