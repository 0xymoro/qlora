export BASE_DIR=/workspace
export WANDB_API_KEY=a028d39551b29cf625bdb6dd4efeac6820948de7
export WANDB_PROJECT=v4-fine-tune
export WANDB_ENTITY=kindroid
export HF_HOME=$BASE_DIR/huggingface
export HF_TOKEN=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

accelerate launch -m axolotl.cli.train 70b-base.yaml

# To upload in case there is an error
# Note that it might upload as pulic.
# huggingface-cli upload Kindroid/v4-llama-8B-Instruct qlora-out --token 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

huggingface-cli upload Kindroid/v4-70B-Base-QLoRA 70b-base-qlora-out/checkpoint-244 checkpoint-244 --token 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
huggingface-cli upload Kindroid/v4-70B-Base-QLoRA 70b-base-qlora-out/checkpoint-488 checkpoint-488 --token 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
huggingface-cli upload Kindroid/v4-70B-Base-QLoRA 70b-base-qlora-out/checkpoint-732 checkpoint-732 --token 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
huggingface-cli upload Kindroid/v4-70B-Base-QLoRA 70b-base-qlora-out/checkpoint-976 checkpoint-776 --token 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
