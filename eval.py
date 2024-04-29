import os.path as osp
import json
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from src.benchmarks import get_qa_dataset, get_semistructured_data
from src.models import get_model
from src.tools.args import merge_args, load_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon", choices=['amazon', 'primekg', 'mag'])
    parser.add_argument(
        "--model", default="VSS", choices=["VSS", "MultiVSS", "LLMReranker"]
    )
    parser.add_argument("--split", default="test")

    # can eval on a subset only
    parser.add_argument("--test_ratio", type=float, default=1.0)

    # for multivss
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--multi_vss_topk", type=int, default=None)
    parser.add_argument("--aggregate", type=str, default="max")

    # for vss, multivss, and llm reranker
    parser.add_argument("--emb_model", type=str, default="text-embedding-ada-002")

    # for llm reranker
    parser.add_argument("--llm_model", type=str, default="gpt-4-1106-preview",
                        help='the LLM to rerank candidates.')
    parser.add_argument("--llm_topk", type=int, default=20)
    parser.add_argument("--max_retry", type=int, default=3)

    # path
    parser.add_argument("--emb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # save prediction
    parser.add_argument("--save_pred", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_args = load_args(
        json.load(open("config/default_args.json", "r"))[args.dataset]
    )
    args = merge_args(args, default_args)

    args.query_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "query")
    args.node_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "doc")
    args.chunk_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "chunk")
    surfix = args.llm_model if args.model == 'LLMReranker' else args.emb_model
    output_dir = osp.join(args.output_dir, "eval", args.dataset, args.model, surfix)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.query_emb_dir, exist_ok=True)
    os.makedirs(args.chunk_emb_dir, exist_ok=True)
    os.makedirs(args.node_emb_dir, exist_ok=True)
    json.dump(vars(args), open(osp.join(output_dir, "args.json"), "w"), indent=4)

    eval_csv_path = osp.join(output_dir, f"eval_results_{args.split}.csv")
    final_eval_path = (
        osp.join(output_dir, f"eval_metrics_{args.split}.json")
        if args.test_ratio == 1.0
        else osp.join(output_dir, f"eval_metrics_{args.split}_{args.test_ratio}.json")
    )

    kb = get_semistructured_data(args.dataset)
    qa_dataset = get_qa_dataset(args.dataset)
    model = get_model(args, kb)

    split_idx = qa_dataset.get_idx_split(test_ratio=args.test_ratio)

    eval_metrics = [
        "mrr",
        "map",
        "rprecision",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "recall@100",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "hit@50",
    ]
    eval_csv = pd.DataFrame(columns=["idx", "query_id", "pred_rank"] + eval_metrics)

    existing_idx = []
    if osp.exists(eval_csv_path):
        eval_csv = pd.read_csv(eval_csv_path)
        existing_idx = eval_csv["idx"].tolist()

    indices = split_idx[args.split].tolist()

    for idx in tqdm(indices):
        if idx in existing_idx:
            continue
        query, query_id, answer_ids, meta_info = qa_dataset[idx]
        pred_dict = model.forward(query, query_id)

        answer_ids = torch.LongTensor(answer_ids)
        result = model.evaluate(pred_dict, answer_ids, metrics=eval_metrics)

        result["idx"], result["query_id"] = idx, query_id
        result["pred_rank"] = torch.LongTensor(list(pred_dict.keys()))[
            torch.argsort(torch.tensor(list(pred_dict.values())), descending=True)[
                :1000
            ]
        ].tolist()

        eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)

        if args.save_pred:
            eval_csv.to_csv(eval_csv_path, index=False)
        for metric in eval_metrics:
            print(
                f"{metric}: {np.mean(eval_csv[eval_csv['idx'].isin(indices)][metric])}"
            )
    if args.save_pred:
        eval_csv.to_csv(eval_csv_path, index=False)
    final_metrics = (
        eval_csv[eval_csv["idx"].isin(indices)][eval_metrics].mean().to_dict()
    )
    json.dump(final_metrics, open(final_eval_path, "w"), indent=4)