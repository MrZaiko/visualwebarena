#!/bin/bash
# set -euo pipefail

###############################################################################
# Unified run script — replaces all scripts/run_* scripts.
#
# Usage:
#   bash scripts/run.sh --website <reddit|classifieds|shopping> \
#                        --model <model_name> \
#                        --result_dir <base_path> \
#                        --instruction_path <path> \
#                        [--start_idx N] [--max_idx N] [--batch_size N] \
#                        [--provider <anthropic|...>] \
#                        [--temperature T] \
#                        [--captioning_model <model>] \
#                        [--observation_type <image_som|accessibility_tree_with_captioner>] \
#                        [--action_set_tag <som|id_accessibility_tree>]
#
# --result_dir is the top-level results folder. The script automatically
# appends /<website>/<model_short_name>[-meta] based on the model and
# whether the instruction_path contains "metatools".
#
# Model name mapping:
#   claude-sonnet-4-5-20250929  → claude
#   gpt-5.1-2025-11-13         → gpt-5
#   openai/gpt-oss-120b        → gpt-oss
#
# Examples (equivalent to the old individual scripts):
#
#   # run_shopping_som.sh → results in .../shopping/gpt-5
#   bash scripts/run.sh --website shopping --model gpt-5.1-2025-11-13 \
#     --result_dir /mnt/nfs/home/abuzakuk/vwa/results \
#     --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json
#
#   # run_reddit_som_claude.sh → results in .../reddit/claude
#   bash scripts/run.sh --website reddit --model claude-sonnet-4-5-20250929 \
#     --result_dir /mnt/nfs/home/abuzakuk/vwa/results \
#     --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
#     --provider anthropic --start_idx 120
#
#   # run_classifieds_som_claude_metatools.sh → results in .../classifieds/claude-meta
#   bash scripts/run.sh --website classifieds --model claude-sonnet-4-5-20250929 \
#     --result_dir /mnt/nfs/home/abuzakuk/vwa/results \
#     --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s_metatools_classifieds.json \
#     --provider anthropic --start_idx 200
###############################################################################

# --------------- defaults ---------------
website=""
model=""
result_dir=""
instruction_path=""
start_idx=0
max_idx=""
batch_size=50
provider=""
temperature=""
captioning_model="Salesforce/blip2-flan-t5-xl"
observation_type="image_som"
action_set_tag="som"

# --------------- parse args ---------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --website)           website="$2";           shift 2 ;;
        --model)             model="$2";             shift 2 ;;
        --result_dir)        result_dir="$2";        shift 2 ;;
        --instruction_path)  instruction_path="$2";  shift 2 ;;
        --start_idx)         start_idx="$2";         shift 2 ;;
        --max_idx)           max_idx="$2";           shift 2 ;;
        --batch_size)        batch_size="$2";        shift 2 ;;
        --provider)          provider="$2";          shift 2 ;;
        --temperature)       temperature="$2";       shift 2 ;;
        --captioning_model)  captioning_model="$2";  shift 2 ;;
        --observation_type)  observation_type="$2";  shift 2 ;;
        --action_set_tag)    action_set_tag="$2";    shift 2 ;;
        --no_captioning_model) captioning_model=""; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --------------- validate required args ---------------
if [[ -z "$website" || -z "$model" || -z "$result_dir" || -z "$instruction_path" ]]; then
    echo "Error: --website, --model, --result_dir, and --instruction_path are required." >&2
    exit 1
fi

# --------------- map model to short name ---------------
case "$model" in
    claude-sonnet-4-5-20250929) model_short="claude" ;;
    gpt-5.1-2025-11-13)        model_short="gpt-5"  ;;
    openai/gpt-oss-120b)       model_short="gpt-oss" ;;
    *)
        # Fallback: use the model name as-is (strip slashes)
        model_short="${model//\//-}"
        ;;
esac

# Append -meta if instruction_path contains "metatools"
if [[ "$instruction_path" == *metatools* ]]; then
    model_short="${model_short}-meta"
fi

# Build full result_dir: <base>/<website>/<model_short>
result_dir="${result_dir%/}/${website}/${model_short}"

# --------------- website-specific defaults ---------------
case "$website" in
    reddit)
        test_config_base_dir="config_files/vwa/test_reddit"
        [[ -z "$max_idx" ]] && max_idx=210
        [[ "$batch_size" -eq 50 ]] && batch_size=30
        [[ -z "$temperature" ]] && temperature=0.2
        ;;
    classifieds)
        test_config_base_dir="config_files/vwa/test_classifieds"
        [[ -z "$max_idx" ]] && max_idx=234
        ;;
    shopping)
        test_config_base_dir="config_files/vwa/test_shopping"
        [[ -z "$max_idx" ]] && max_idx=466
        # Shopping scripts didn't use captioning_model
        [[ "$captioning_model" == "Salesforce/blip2-flan-t5-xl" ]] && captioning_model=""
        ;;
    *)
        echo "Error: --website must be one of: reddit, classifieds, shopping" >&2
        exit 1
        ;;
esac

# --------------- build optional flags ---------------
optional_flags=()
[[ -n "$provider" ]]          && optional_flags+=(--provider "$provider")
[[ -n "$temperature" ]]       && optional_flags+=(--temperature "$temperature")
[[ -n "$captioning_model" ]]  && optional_flags+=(--captioning_model "$captioning_model")

# --------------- reset function ---------------
reset_website() {
    case "$website" in
        reddit)
            bash scripts/reset_reddit.sh
            ;;
        classifieds)
            curl -X POST "$CLASSIFIEDS/index.php?page=reset" -d "token=4b61655535e7ed388f0d40a93600254c"
            ;;
        shopping)
            bash scripts/reset_shopping.sh
            ;;
    esac
}

# --------------- main loop ---------------
end_idx=$((start_idx + batch_size))
if [[ $end_idx -gt $max_idx ]]; then
    end_idx=$max_idx
fi

while [[ $start_idx -le $max_idx ]]; do
    reset_website
    bash prepare.sh

    uv run run.py \
        --instruction_path "$instruction_path" \
        --test_start_idx "$start_idx" \
        --test_end_idx "$end_idx" \
        --model "$model" \
        --experiment_name "VWA-$model-$instruction_path" \
        --wandb_project "AWO" \
        --wandb_entity "sabuzakuk-epfl" \
        --result_dir "$result_dir" \
        --test_config_base_dir "$test_config_base_dir" \
        --repeating_action_failure_th 5 \
        --viewport_height 2048 \
        --max_obs_length 3840 \
        --eval_captioning_model_device cuda \
        --action_set_tag "$action_set_tag" \
        --observation_type "$observation_type" \
        "${optional_flags[@]}"

    start_idx=$((start_idx + batch_size))
    end_idx=$((start_idx + batch_size))
    if [[ $end_idx -gt $max_idx ]]; then
        end_idx=$max_idx
    fi
done
