export BASE_DIR=/workspace
export WANDB_API_KEY=a028d39551b29cf625bdb6dd4efeac6820948de7
export WANDB_PROJECT=v4-fine-tune
export WANDB_ENTITY=kindroid
export HF_HOME=$BASE_DIR/huggingface
export HF_TOKEN=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

accelerate launch -m axolotl.cli.train 70b-instruct.yaml

huggingface-cli upload Kindroid/v4-70B-Instruct-QLoRA 70b-instruct-qlora-out/checkpoint-241 checkpoint-241 --token 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
huggingface-cli upload Kindroid/v4-70B-Instruct-QLoRA 70b-instruct-qlora-out/checkpoint-482 checkpoint-482 --token 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
huggingface-cli upload Kindroid/v4-70B-Instruct-QLoRA 70b-instruct-qlora-out/checkpoint-723 checkpoint-723 --token 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
huggingface-cli upload Kindroid/v4-70B-Instruct-QLoRA 70b-instruct-qlora-out/checkpoint-964 checkpoint-964 --token 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
