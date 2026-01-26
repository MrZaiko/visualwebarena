#!/bin/bash
### This script runs the GPT-4V + SoM models on the entire VWA shopping test set.

model="claude-sonnet-4-5-20250929"
result_dir="/mnt/nfs/home/abuzakuk/vwa/results/shopping/claude"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"

# Define the batch size variable
batch_size=50

# Define the starting and ending indices
start_idx=300
end_idx=$((start_idx + batch_size))
max_idx=466

# Loop until the starting index is less than or equal to max_idx.
while [ $start_idx -le $max_idx ]
do
    bash scripts/reset_shopping.sh
    bash prepare.sh
    uv run run.py \
     --instruction_path $instruction_path \
     --test_start_idx $start_idx \
     --test_end_idx $end_idx \
     --model $model \
     --result_dir $result_dir \
     --test_config_base_dir=config_files/vwa/test_shopping \
     --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
     --eval_captioning_model_device cuda \
     --provider anthropic \
     --action_set_tag som  --observation_type image_som

    # Increment the start and end indices by the batch size
    start_idx=$((start_idx + batch_size))
    end_idx=$((end_idx + batch_size))

    # Ensure the end index does not exceed 466 in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
done
