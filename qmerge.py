# Modified from https://gist.githubusercontent.com/ChrisHayduk/1a53463331f52dca205e55982baf9930/raw/438ab25f05a8e1dd3c384b81fad38c6101c98be9/merge_qlora_with_quantized_model.py
import argparse
import torch
import peft
import json
import shutil
from peft.utils import _get_submodules
import os
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig #CodeLlamaTokenizer
import gc
import copy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--peft", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()

#Note: careful on devices. Usually CPU for dequant, but if it is in CPU after dequant it will hang if the peft/etc are on "auto" most likely on GPUs.
def dequantize_model(model, tokenizer, to, dtype=torch.bfloat16, device="auto"): 
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to after dequantization
    """
    original_device = device  # Store the original device configuration
    
    if os.path.exists(to):
        return AutoModelForCausalLM.from_pretrained(to, torch_dtype=torch.bfloat16, device_map=original_device)
    
    os.makedirs(to, exist_ok=True)
    cls = torch.nn.Linear  # Example class for the linear layer, replace with your specific class
    
    with torch.no_grad():
        model.to('cpu')  # Ensure model is on CPU for dequantization
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                # Simulated dequantization process
                weights = module.weight.data.float()  # Convert weights to float
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None)
                new_module.weight = torch.nn.Parameter(weights)
                parent_name, child_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, new_module)

        model.is_loaded_in_4bit = False
        print("Saving dequantized model...")
        model.save_pretrained(to)
        tokenizer.save_pretrained(to)
        
        # Modify configuration to remove quantization entries
        config_path = os.path.join(to, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
            config_data.pop("quantization_config", None)
            config_data.pop("pretraining_tp", None)
            with open(config_path, 'w') as config_file:
                json.dump(config_data, config_file, indent=2)

    # After dequantization, load the model to the specified device
    model = AutoModelForCausalLM.from_pretrained(to, torch_dtype=dtype, device_map=original_device)
    return model


def main():
    args = get_args()
    model_path = args.base
    adapter_path = args.peft
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"Loading base model: {model_path}")
    model = None
    if os.path.exists(f"{model_path}-dequantized"):
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}-dequantized")
        model = AutoModelForCausalLM.from_pretrained(
            f"{model_path}-dequantized",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = dequantize_model(model, tokenizer, to=f"{model_path}-dequantized")
    model = PeftModel.from_pretrained(model=model, model_id=adapter_path)
    model = model.merge_and_unload()
    print("Successfully loaded and merged model, saving...")
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size='10GB')
    tokenizer.save_pretrained(args.out)
    config_data = json.loads(open(os.path.join(args.out, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(args.out, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))
    print(f"Model saved: {args.out}")
    if args.push:
        print(f"Saving to hub ...")
        model.push_to_hub(args.out, use_temp_dir=False)
        tokenizer.push_to_hub(args.out, use_temp_dir=False)
        print("Model successfully pushed to hf.")

if __name__ == "__main__":
    main()
