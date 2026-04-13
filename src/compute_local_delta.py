import os
import json
import random

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

random.seed(42)

backbone_model_path = ''
model_path_list = [
]
top_k = 1

@torch.no_grad()
def svd_then_recover(model_path, top_k, backbone_state_dict):
    print(model_path)

    model_name = 'svd_topk_{}.bin'.format(str(top_k))
    tgt_folder = os.path.join(model_path, 'last_delta_svd_matrix')
    tgt_path = os.path.join(tgt_folder, model_name)
    os.makedirs(tgt_folder, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    state_dict = model.state_dict()
    processed_state_dict = {}
    size_to_tensor = {}
    size_to_name = {}

    @torch.no_grad()
    def add_tensor(tensor, nam):
        nonlocal size_to_tensor, size_to_name
        if (tensor.size(0) != 1 and tensor.size(-1) != 1):
            tensor = tensor.unsqueeze(-1)
        raw_tensor_size = tensor.size()
        tensor_size = '{};{}'.format(raw_tensor_size[0], raw_tensor_size[1])
        if (tensor_size not in size_to_tensor):
            size_to_tensor[tensor_size] = []
            size_to_name[tensor_size] = []
        size_to_tensor[tensor_size].append(tensor)
        size_to_name[tensor_size].append(nam)

    for nam, raw_param in tqdm(state_dict.items()):
        print(nam)
        param = raw_param - backbone_state_dict[nam]
        if len(param.size()) < 2:
            print('\t[No SVD]', param.size())
            add_tensor(param, nam)
        else:
            U, S, Vh = torch.linalg.svd(param, full_matrices=False)
            threshold = torch.topk(S, top_k).values[-1]
            S = torch.where(S >= threshold, S, 0.0)
            ut = U[:, :1]
            st = torch.tensor(threshold).reshape(1, 1)
            vt = Vh[:1, :]
            print('\t', ut.size(), st.size(), vt.size())

            add_tensor(ut, f'{nam}-ut')
            add_tensor(st, f'{nam}-st')
            add_tensor(vt, f'{nam}-vt')

    size_list = size_to_tensor.keys()
    final_state_dict = {}
    for k in size_list:
        first_dim = int(k.split(';')[0].strip())
        second_dim = int(k.split(';')[-1].strip())
        if second_dim == 1:
            tensor_matrix = torch.cat(size_to_tensor[k], dim=-1)
        else:
            tensor_matrix = torch.cat(size_to_tensor[k], dim=0)
        final_state_dict[k] = tensor_matrix
        print(k, ':', len(size_to_tensor[k]), tensor_matrix.size())

    torch.save(final_state_dict, tgt_path)
    with open(os.path.join(tgt_folder, f'size_to_name-{model_name}.json'), 'w') as fout:
        json.dump(size_to_name, fout, indent=4)

if __name__ == '__main__':
    with torch.no_grad():
        backbone_model = AutoModelForCausalLM.from_pretrained(backbone_model_path)
        backbone_state_dict = backbone_model.state_dict()
        last_state_dict = backbone_state_dict
        for model_path in model_path_list:
            # svd_then_recover(model_path, top_k, backbone_state_dict)
            svd_then_recover(model_path, top_k, last_state_dict)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            last_state_dict = model.state_dict()