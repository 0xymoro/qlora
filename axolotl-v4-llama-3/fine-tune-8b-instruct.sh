export BASE_DIR=/workspace
export WANDB_API_KEY=a028d39551b29cf625bdb6dd4efeac6820948de7
export WANDB_PROJECT=v4-fine-tune
export WANDB_ENTITY=kindroid
export HF_HOME=$BASE_DIR/huggingface
export HF_TOKEN=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

accelerate launch -m axolotl.cli.train 8b-instruct.yaml
