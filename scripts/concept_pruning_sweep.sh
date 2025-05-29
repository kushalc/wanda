#!/bin/bash -e

model="gemma-2-2b"
sae="20-gemmascope-res-16k"
OUTPUT_DIR="outputs/`date +%Y-%m-%d-%H-%M-%S`"

# model="meta-llama/Llama-3.1-8B"
# sae="25-llamascope-res-32k"

# baseline
for prune_method in magnitude wanda; do
    for sparsity_ratio in 0.500 0.250 0.125 0.064 0.032 0.016 0.008; do
        python main.py \
            --model $model \
            --sae $sae \
            --prune_method $prune_method \
            --sparsity_ratio $sparsity_ratio \
            --activation_dataset c4 \
            --eval hallucination zero_shot perplexity \
            --nsamples_eval 64 \
            --output_dir $OUTPUT_DIR \
            --device mps
    done
done

# FIXME: Commented out since pdb doesn't exit unclean.
# for K in 0 5 10 20; do
#     python main.py \
#         --model $model \
#         --sae $sae \
#         --prune_method concept_noise \
#         --concept_noise_k $K \
#         --activation_dataset c4 \
#         --save outputs/$model/K$K-L0 \
#         --eval_hallucination_metrics \
#         --device mps
# done

# for L in 0 5 10 20; do
#     python main.py \
#         --model $model \
#         --sae $sae \
#         --prune_method concept_prune \
#         --concept_prune_l $L \
#         --activation_dataset c4 \
#         --save outputs/$model/K0-L$L \
#         --eval_hallucination_metrics
#         --device mps
# done
