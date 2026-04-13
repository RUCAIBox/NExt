import os
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


top_k = 1
backbone_model_path = ""
model_path_list = [
    "",
]

@torch.no_grad()
def merge_lora(model_path):
    print(model_path)

    merged_model_path = os.path.join(model_path, 'hf_model')

    model = AutoModelForCausalLM.from_pretrained(backbone_model_path)
    lora_model = PeftModel.from_pretrained(model, model_id=os.path.join(model_path, 'actor', 'lora_adapter'))
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_path)

    tokenizer = AutoTokenizer.from_pretrained(backbone_model_path)
    tokenizer.save_pretrained(merged_model_path)

if __name__ == '__main__':
    with torch.no_grad():
        for model_path in tqdm(model_path_list):
            merge_lora(model_path)