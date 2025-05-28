import argparse
import logging
import os
import pprint
from importlib.metadata import version

import numpy as np
import torch
from lib.eval import eval_ppl, eval_zero_shot
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
    if sae_name is not None:
        model = HookedSAETransformer.from_pretrained_no_processing(model_name, **kwargs).to(device)

        pretrained_df = load_pretrained_saes()
        release, sae_id = pretrained_df.loc[(model_name, sae_name), ["hf_release", "hf_sae_id"]]
        sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        model.add_sae(sae)
        tokenizer = model.tokenizer

        logging.info("Attached SAEs: %s", pprint.pformat(model.acts_to_saes, compact=True))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # NOTE: Modern models have significantly longer context windows and C4 doesn't have enough
    # examples that exceed these.
    # model.seqlen = model.config.max_position_embeddings
    model.seqlen = 2048
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='HuggingFace or SAELens transformer')
    parser.add_argument("--sae", type=str, default=None, help="Pretrained SAE")

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
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
        ],
    )
    parser.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/prune_llm"), type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--device', type=str, default="cuda", help='Default device to use.')
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--concept_noise_k", type=int, default=0, help="Number of concepts to apply noise")
    parser.add_argument("--concept_prune_l", type=int, default=0, help="Number of concepts to prune")
    parser.add_argument("--activation_dataset", type=str, default="c4", help="Dataset for concept activations")
    parser.add_argument("--eval_hallucination_metrics", action="store_true", help="Compute hallucination metrics")
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    logging.info(f"loading llm model {args.model}")
    model, tokenizer = get_llm(args.model, args.cache_dir, args.device, sae_name=args.sae)
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

    ################################################################
    logging.info("*"*30)
    sparsity_ratio = check_sparsity(model)
    logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
    logging.info("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    logging.info(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_hallucination_metrics:
        from lib.eval import evaluate_hallucination
        metrics = evaluate_hallucination(args, model, tokenizer, device)
        logging.info("hallucination metrics %s", metrics)

    if args.eval_zero_shot:
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        logging.info("********************************")
        logging.info("zero_shot evaluation results")
        logging.info(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
