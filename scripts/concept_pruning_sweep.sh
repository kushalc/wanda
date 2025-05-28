#!/bin/bash

model_base="gemma-2-2b"
model_full="meta-llama/Llama-3.1-8B"

for K in 0 5 10 20; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model $model_base \
        --prune_method concept_noise \
        --sae_model gemma-2-2b/20-gemmascope-res-16k \
        --concept_noise_k $K \
        --activation_dataset c4 \
        --save out/concept_noise/gemma/K${K}/ \
        --eval_hallucination_metrics
done

for L in 0 5 10 20; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model $model_full \
        --prune_method concept_prune \
        --sae_model 25-llamascope-res-32k \
        --concept_prune_l $L \
        --activation_dataset c4 \
        --save out/concept_prune/llama3/L${L}/ \
        --eval_hallucination_metrics
done
