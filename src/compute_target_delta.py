import os
import json
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

random.seed(42)

backbone_model_path = ''
model_template = ''
long_inter = 50

max_step = 150
min_step = 10
save_inter = 10

model_path_list = list(range(min_step, max_step + 1, save_inter))
print('Step List:', model_path_list, '\n')

top_k = 1

@torch.no_grad()
def svd_then_recover(ckpt_idx, top_k):
    if (ckpt_idx + long_inter > max_step):
        return

    model_path = model_template.format(ckpt_idx)
    long_ckpt_idx = ckpt_idx + long_inter
    long_model_path = model_template.format(long_ckpt_idx)

    print('Current Model Path: ', model_path)
    print('Long Model Path: ', long_model_path)

    model_name = 'svd_topk_{}.bin'.format(str(top_k))
    tgt_folder = os.path.join(model_path, 'long_delta_svd_matrix')
    tgt_path = os.path.join(tgt_folder, model_name)
    os.makedirs(tgt_folder, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    state_dict = model.state_dict()

    long_model = AutoModelForCausalLM.from_pretrained(long_model_path)
    long_state_dict = long_model.state_dict()

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
        param = long_state_dict[nam] - raw_param
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
        for ckpt_idx in model_path_list:
            svd_then_recover(ckpt_idx, top_k)