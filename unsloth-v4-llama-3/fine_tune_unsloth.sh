export BASE_DIR=/workspace
export WANDB_API_KEY=a028d39551b29cf625bdb6dd4efeac6820948de7
export WANDB_PROJECT=v4-test
export WANDB_ENTITY=kindroid
export HF_HOME=$BASE_DIR/huggingface
export HF_TOKEN=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

python fine_tune_unsloth.py \
	--model-name unsloth/llama-3-8b-bnb-4bit \
	--max-seq-length 8192 \
	--parquet-path kindroid-v4-sft.parquet \
	--output-dir $BASE_DIR/$WANDB_PROJECT \
	--hf-kindroid-model-name llama-3-8b-v4-test
