#!/bin/bash

# Define an array of the model sizes you want to evaluate.
# You can add or remove models from this list.
# The format should be "model-name:size"
models=(
    "gemma3:1b"
    "gemma3:4b"
    "gemma3:12b"
#    "gemma3:27b"
#    "deepseek-r1:1.5b"
#    "deepseek-r1:8b"
    "deepseek-r1:14b"
#    "deepseek-r1:32b"
)

# Loop through each model in the array
for model in "${models[@]}"
do
    echo "Starting evaluation for model: $model"

    # The lm_eval command. We use variable substitution for the model name.
    # Note how we construct the output path and the model_args string dynamically.
    lm_eval --model local-chat-completions \
        --tasks hotpotqa_fullwiki_include_context,hotpotqa_fullwiki_include_context \
        --log_samples \
        --output_path "./results/$model/" \
        --write_out \
        --trust_remote_code \
        --model_args "model=$model,base_url=http://localhost:11434/v1/chat/completions/,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192,apply_chat_template=True,eos_string='<end_of_turn>'" \
        --apply_chat_template \
        --include_path lm_eval/tasks/hotpotqa_fullwiki

    # Check the exit status of the previous command.
    # If it's not 0, something went wrong.
    if [ $? -ne 0 ]; then
        echo "Error: lm_eval failed for model $model. Stopping script."
        # You could also use 'continue' here to skip to the next model
        # instead of exiting the script entirely.
        continue
    fi

    echo "Evaluation for $model completed successfully."
    echo "----------------------------------------"
done

echo "All evaluations have been completed."