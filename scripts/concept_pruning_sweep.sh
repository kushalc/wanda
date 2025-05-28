#!/bin/bash -e

model="gemma-2-2b"
sae="20-gemmascope-res-16k"

# model="meta-llama/Llama-3.1-8B"
# sae="25-llamascope-res-32k"

K=0
# python -m pdb -c continue main.py \
python main.py \
    --model $model \
    --sae $sae \
    --prune_method concept_noise \
    --concept_noise_k $K \
    --activation_dataset c4 \
    --save outputs/$model/K$K-L0 \
    --eval_hallucination_metrics \
    --device mps

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
