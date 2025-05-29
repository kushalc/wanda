import argparse
import logging
import os
import pickle
import pprint
import time
from importlib.metadata import version
from pathlib import Path

import lib.eval
import numpy as np
import pandas as pd
import torch
from lib.prune import (check_sparsity, find_layers, prune_ablate,
                       prune_concept_noise, prune_concept_prune,
                       prune_magnitude, prune_sparsegpt, prune_wanda)
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utilities.etl import load_pretrained_saes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(funcName)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logging.info('torch: %s', version('torch'))
logging.info('transformers: %s', version('transformers'))
logging.info('accelerate: %s', version('accelerate'))
logging.info('# of gpus: %s', torch.cuda.device_count())


def get_llm(model_name, cache_dir, device, sae_name=None):
    kwargs = {
        "torch_dtype": torch.float16,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        # "device_map": device,  # NOTE: HookedSAETransformer gets confused and loads into both mps and cpu
    }
    meta = {}
    if sae_name is not None:
        model = HookedSAETransformer.from_pretrained_no_processing(model_name, **kwargs).to(device)

        pretrained_df = load_pretrained_saes()
        release, sae_id = pretrained_df.loc[(model_name, sae_name), ["hf_release", "hf_sae_id"]]
        sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        model.add_sae(sae)
        tokenizer = model.tokenizer
        meta["sae"] = sae

        logging.info("Attached SAEs: %s", pprint.pformat(model.acts_to_saes, compact=True))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # NOTE: Modern models have significantly longer context windows and C4 doesn't have enough
    # examples that exceed these.
    # model.seqlen = model.config.max_position_embeddings
    model.seqlen = 2048
    return model, tokenizer, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='HuggingFace or SAELens transformer')
    parser.add_argument("--sae", type=str, default=None, help="Pretrained SAE")

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--nsamples_eval', type=int, default=64, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=[
        "magnitude",
        "wanda",
        "sparsegpt",
        "ablate_mag_seq",
        "ablate_wanda_seq",
        "ablate_mag_iter",
        "ablate_wanda_iter",
        "concept_noise",
        "concept_prune",
        "search",
    ])
    parser.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/prune_llm"), type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--device', type=str, default="cuda", help='Default device to use.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to save results.')

    parser.add_argument("--concept_noise_k", type=int, default=0, help="Number of concepts to apply noise")
    parser.add_argument("--concept_prune_l", type=int, default=0, help="Number of concepts to prune")
    parser.add_argument("--activation_dataset", type=str, default="c4", help="Dataset for concept activations")
    parser.add_argument("--eval", nargs="+", choices=["perplexity", "zero_shot", "hallucination"], help="Calculate chosen metrics")
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    started_at = time.time()
    logging.info("Loading llm model %s", args.model)
    model, tokenizer, meta = get_llm(args.model, args.cache_dir, args.device, sae_name=args.sae)
    model.eval()

    device = torch.device(args.device)
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logging.info("use device %s", device)

    if args.sparsity_ratio != 0:
        logging.info("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "concept_noise":
            prune_concept_noise(args, model, tokenizer, device)
        elif args.prune_method == "concept_prune":
            prune_concept_prune(args, model, tokenizer, device)

    _, sparsity_meta = check_sparsity(model)
    metrics = {
        "args": vars(args),
        "executed_at": pd.Timestamp.now(),
        "duration": time.time() - started_at,
        "sparsity": sparsity_meta,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = args.output_dir / metrics["executed_at"].strftime("%Y-%m-%d-%H-%M-%S.pkl")
    logging.info("Saving results to %s", metrics_path)
    pickle.dump(metrics, open(metrics_path, "wb"))

    for name in args.eval:
        try:
            started_at = time.time()
            name = f"lib.eval.eval_{name}"
            logging.info("Starting eval=%s", name)
            metrics[name] = eval(name)(args, model, tokenizer, sae=meta.get("sae"))
            metrics[name]["duration"] = time.time() - started_at
        except:
            logging.warning("Couldn't run eval: %s", name, exc_info=True)
        pickle.dump(metrics, open(metrics_path, "wb"))


if __name__ == '__main__':
    main()
