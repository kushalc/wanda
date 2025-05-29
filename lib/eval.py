# Import necessary modules
import logging
import time
from collections import defaultdict

import lm_eval
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

TASK_LIST = ["boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"]


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_perplexity(args, model, tokenizer, batch_size=32, dataset="wikitext2", **kwargs):
    logging.info(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = _eval_ppl_wikitext(model, testloader, 1, args.device, nsamples=args.nsamples_eval)
    return {
        "perplexity": ppl_test,
    }

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset


def _eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
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


def _eval_ppl_wikitext(model, testenc, bs=1, device=None, nsamples=64):  # NOTE: bs=1 is different than batch_size above...
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


def eval_zero_shot(args, model, tokenizer, num_fewshot=0, batch_size=32, use_accelerate=False, **kwargs):
    model_args = [f"cache_dir={args.cache_dir}"]
    if use_accelerate:
        model_args += ["use_accelerate=True"]

    # FIXME: Trying to hack in a bunch of stuff to make HookedSAETransformer <=> HFLM compatible, but guessing
    # there's more nontrivial stitching we have to do.
    mgr = tasks.TaskManager()
    results = evaluator.simple_evaluate(model=HookedTransformerLM(model, tokenizer, device=args.device),
                                        model_args=",".join(model_args), tasks=mgr.match_tasks(TASK_LIST),
                                        num_fewshot=num_fewshot, check_integrity=False, limit=args.nsamples_eval,
                                        batch_size=batch_size, device=args.device, task_manager=mgr,
                                        use_cache=None)  # NOTE: Don't want corruption-unaware cache
    return results


def eval_hallucination(args, model, tokenizer, activation_threshold=1.0, sae=None, batch_size=32, **kwargs):
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
    sampled_idx = np.random.choice(truthful_df["core_prompt_idx"].unique(), size=args.nsamples_eval, replace=False)
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
                               truncation=True, max_length=model.seqlen).to(args.device)

            # FIXME: Either need to re-implement this method or provide parallel implementation for
            # prune_magnitude and prune_wanda. Do latter for stricter comparability.
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


class HookedTransformerLM(lm_eval.api.model.LM):
    """
    Wrapper to use a TransformerLens HookedTransformer with lm_eval.
    Implements loglikelihood, loglikelihood_rolling, and generate_until.
    """

    def __init__(self, model, tokenizer, device=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else self.tokenizer.get_vocab().__len__()
        self._eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None

    @property
    def max_length(self):
        return getattr(self.model, 'seqlen', 2048)

    @property
    def eot_token_id(self):
        return self._eos_token_id

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        # requests: List of dicts with 'context' and 'continuation' keys
        res = []
        for req in requests:
            context, continuation = req.args
            input_str = context + continuation
            input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids.to(self.device)
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids)[0] if hasattr(self.model(input_ids), 'logits') else self.model(input_ids)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            cont_len = input_ids.shape[1] - context_ids.shape[1]
            cont_ids = input_ids[0, -cont_len:]
            cont_logprobs = log_probs[0, -cont_len-1:-1].gather(1, cont_ids.unsqueeze(-1)).squeeze(-1)
            total_logprob = cont_logprobs.sum().item()
            is_greedy = (cont_logprobs.argmax(dim=-1) == cont_ids).all().item()
            res.append((total_logprob, is_greedy))
        return res

    def loglikelihood_rolling(self, requests):
        # For perplexity: requests is a list of strings
        res = []
        for string in requests:
            input_ids = self.tokenizer(string, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids)[0] if hasattr(self.model(input_ids), 'logits') else self.model(input_ids)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            labels = input_ids[:, 1:]
            log_probs = log_probs[:, :-1, :]
            nll = torch.nn.functional.nll_loss(log_probs.reshape(-1, log_probs.size(-1)), labels.reshape(-1), reduction='sum')
            res.append((-nll.item(), input_ids.shape[1] - 1))
        return res

    def generate_until(self, requests):
        # requests: list of dicts with 'context', 'until', 'max_length', 'eos_token_id'
        results = []
        for req in requests:
            context = req["context"]
            until = req.get("until", [])
            max_length = req.get("max_length", 20)
            eos_token_id = req.get("eos_token_id", self.eot_token_id)
            # Generate tokens
            output_ids = self.generate(context, max_length, eos_token_id)
            output_text = self.tok_decode(output_ids)
            # Truncate at any stop sequence in 'until'
            for stop_seq in until:
                idx = output_text.find(stop_seq)
                if idx != -1:
                    output_text = output_text[:idx]
                    break
            results.append(output_text)
        return results
