# Import necessary modules
import fnmatch
import logging
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from lm_eval import evaluator, tasks
from tqdm import tqdm

from utilities.etl import load_truthfulqa

# Import get_loaders function from data module within the same directory
from .data import get_loaders


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0"), nsamples=64):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    logging.info(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device, nsamples=nsamples)
    return ppl_test

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset


def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    logging.info(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs)):
        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        output = model(inputs)
        lm_logits = output.logits if hasattr(output, 'logits') else output

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset


def eval_ppl_wikitext(model, testenc, bs=1, device=None, nsamples=64):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    if nsamples is None:
        nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs)):
        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        output = model(inputs)
        lm_logits = output.logits if hasattr(output, 'logits') else output

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"],
                   num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens
    )

    return results


def evaluate_hallucination(model, sae, tokenizer, device, activation_threshold=1.0,
                           nsamples=64, batch_size=32):
    """
    Evaluate hallucination on TruthfulQA using SAE concepts and logprobs, batching to save memory.
    Args:
        model: HookedSAETransformer
        sae: SAE module (with .encode())
        tokenizer: tokenizer
        device: torch.device
        activation_threshold: threshold for concept activation (default: 0.0)
        batch_size: number of prompts per batch
    Returns:
        dict with accuracy and concept activation stats
    """
    truthful_df = load_truthfulqa(mc=True, preset="null")
    sampled_idx = np.random.choice(truthful_df["core_prompt_idx"].unique(), size=nsamples, replace=False)
    truthful_df = truthful_df[truthful_df["core_prompt_idx"].isin(sampled_idx)]

    # Prepare all prompt texts and answer types
    prompts = truthful_df["_prompt_text"].tolist()
    answer_types = truthful_df["AnswerType"].tolist()

    all_logits = []
    all_concepts = []
    # Batch processing
    for start in tqdm(range(0, len(prompts), batch_size)):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        with torch.no_grad():
            tokens = tokenizer(batch_prompts, return_tensors="pt", padding="longest",
                               truncation=True, max_length=model.seqlen).to(device)
            hook_name = sae.cfg.hook_name + ".hook_sae_acts_post"
            logits, cache = model.run_with_cache(tokens.input_ids, return_type="logits",
                                                 return_cache_object=True, names_filter=[hook_name])
            all_concepts.append(cache[hook_name].cpu())
            all_logits.append(logits.cpu())

    # Concatenate all batches, find max sequence length across all batches
    max_seq_len = max(x.shape[1] for x in all_logits)

    def pad_to_max_seq(x):
        return nnf.pad(x, (0, 0, 0, max_seq_len - x.shape[1])) if max_seq_len > x.shape[1] else x
    logits = torch.cat([pad_to_max_seq(x) for x in all_logits], dim=0)
    sae_acts = torch.cat([pad_to_max_seq(x) for x in all_concepts], dim=0)

    # Compute logprob and concept_acts for each row in truthful_df
    truthful_df = truthful_df.assign(logprob=pd.NA, concepts=None)
    for i, (idx, row) in enumerate(truthful_df.iterrows()):
        prompt_len = len(tokenizer(row["_base_text"], return_tensors="pt").input_ids[0])
        answer_tokens = tokenizer(row["_prompt_text"], return_tensors="pt").input_ids[0][prompt_len:]
        answer_range = range(prompt_len, prompt_len+len(answer_tokens))

        truthful_df.at[idx, "logprob"] = nnf.log_softmax(logits[i, answer_range], dim=-1)[range(len(answer_tokens)), answer_tokens].sum().item()

        activations = np.where((sae_acts[i, answer_range, :].cpu().numpy() > activation_threshold).any(axis=0))[0]
        truthful_df.at[idx, "concepts"] = activations.tolist()

    # Group by core_prompt_idx, get all answers for each question
    # Compute is_group_correct for each group
    def group_correct(group):
        if not (group["AnswerType"] == "Correct").any() or not (group["AnswerType"] == "Incorrect").any():
            return False
        max_correct = group.loc[group["AnswerType"] == "Correct", "logprob"].max()
        max_incorrect = group.loc[group["AnswerType"] == "Incorrect", "logprob"].max()
        return max_correct > max_incorrect
    truthful_df["correct"] = truthful_df.groupby("core_prompt_idx") \
                                        .apply(group_correct) \
                                        .reindex(truthful_df["core_prompt_idx"]) \
                                        .values

    # Vectorized concept activation analysis
    correct_mask = truthful_df["correct"] == True
    incorrect_mask = truthful_df["correct"] == False

    def _aggregate_concepts(mask):
        return truthful_df.loc[mask] \
                          .explode("concepts") \
                          .groupby("concepts") \
                          .agg({"logprob": "mean", "core_prompt_idx": "nunique", "Answer": "count"}) \
                          .rename(columns={"core_prompt_idx": "nunique", "Answer": "count"}) \
                          .sort_values("nunique", ascending=False)
    concepts_df = pd.concat([_aggregate_concepts(correct_mask), _aggregate_concepts(incorrect_mask)], keys=["Correct", "Incorrect"], axis=1)
    concepts_df["score"] = (concepts_df[("Correct", "nunique")].fillna(0) + 1) / \
                           (concepts_df.xs("nunique", axis=1, level=1).fillna(0).sum(axis=1) + 2)  # pseudocounts
    concepts_df = concepts_df.sort_values("score", ascending=False)

    accuracy = correct_mask.groupby(truthful_df["core_prompt_idx"]).first().mean()
    printable_cols = ["core_prompt_idx", "Question", "Answer", "AnswerType", "concepts", "logprob", "correct"]
    logging.info("Calculated TruthfulQA accuracy=%.3f (sample=%d):\n%s", accuracy, truthful_df["core_prompt_idx"].nunique(),
                 truthful_df.to_string(max_colwidth=50, line_width=250, columns=printable_cols))

    return {
        "accuracy": accuracy,
        "total": truthful_df["core_prompt_idx"].nunique(),
        "truthful_df": truthful_df[printable_cols],
        "concepts_df": concepts_df,
    }
