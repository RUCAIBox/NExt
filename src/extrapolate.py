import argparse
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

class MatrixPredict(nn.Module):
    def __init__(self, dim, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.model_delta = nn.Sequential(
            nn.Linear(dim,256),
            nn.Linear(256,dim),
            nn.ReLU(),
            nn.Linear(dim,256),
            nn.Linear(256,dim),
        )
        self.model_last_delta = nn.Sequential(
            nn.Linear(dim,256),
            nn.Linear(256,dim),
            nn.ReLU(),
            nn.Linear(dim,256),
            nn.Linear(256,dim),
        )
        self.model_combine = nn.Sequential(
            nn.Linear(dim * 2,256),
            nn.Linear(256,dim * 2),
            nn.ReLU(),
            nn.Linear(dim*2,256),
            nn.Linear(256,dim),
        )

    def forward(self, delta_ckpt, last_delta_ckpt, label_ckpt=None):
        output_delta_ckpt = self.model_delta(delta_ckpt)
        output_last_delta_ckpt = self.model_last_delta(last_delta_ckpt)
        combine_ckpt = torch.cat([output_delta_ckpt, output_last_delta_ckpt],dim=-1)
        output_ckpt = self.model_combine(combine_ckpt)
        if (label_ckpt is not None):
            if (self.loss_func == '12'):
                loss = torch.square(output_ckpt - label_ckpt).sum()
            elif (self.loss_func == '11'):
                loss = torch.abs(output_ckpt - label_ckpt).sum()
            else:
                raise(ValueError)
            return loss
        else:
            return output_ckpt


def load_train_ckpt(args):
    step2ckpt_delta = {}
    step2ckpt_last_delta = {}
    step2ckpt_long_delta = {}
    
    for folder_name in os.listdir(args.model_folder):
        if ('global_step_' not in folder_name):
            continue
    
        ckpt_step = int(folder_name.split('global_step_')[-1].strip())
        if (ckpt_step <= args.max_steps):
            model_name = 'svd_topk_{}.bin'.format(str(args.svd_topk))

            ckpt_path = os.path.join(args.model_folder, folder_name, args.default_dir, "delta_svd_matrix", model_name)
            step2ckpt_delta[ckpt_step] = ckpt_path
            ckpt_path = os.path.join(args.model_folder, folder_name, args.default_dir, "last_delta_svd_matrix", model_name)
            step2ckpt_last_delta[ckpt_step] = ckpt_path

            name_path = os.path.join(args.model_folder, folder_name, args.default_dir, args.delta_name, f'size_to_name-{model_name}.json')
            with open(name_path, 'r') as fin:
                size_to_name = json.load(fin)
    
    sorted_step2ckpt_delta = sorted(step2ckpt_delta.items(), key=lambda x:x[0])
    ckpts_delta = []
    for _, ckpt_path in sorted_step2ckpt_delta:
        cur_ckpt= torch.load(ckpt_path)
        ckpts_delta.append(cur_ckpt)
    
    sorted_step2ckpt_last_delta = sorted(step2ckpt_last_delta.items(), key=lambda x:x[0])
    ckpts_last_delta = []
    for _, ckpt_path in sorted_step2ckpt_last_delta:
        cur_ckpt= torch.load(ckpt_path)
        ckpts_last_delta.append(cur_ckpt)
    
    print('Total Loading {} CKPTs for Global Delta.'.format(len(ckpts_delta)))
    print('Total Loading {} CKPTs for  Local Delta.'.format(len(ckpts_last_delta)))

    print('Size List:')
    size_list = []
    for sz in ckpts_delta[0].keys():
        first_dim =int(sz.split(';')[0].strip())
        second_dim = int(sz.split(';')[-1].strip())
        size_list.append([first_dim, second_dim])
        print('    ({}, {})'.format(first_dim, second_dim))
    
    model = AutoModelForCausalLM.from_pretrained(args.backbone_model_path)
    backbone_state_dict = model.state_dict()
    
    return ckpts_delta, ckpts_last_delta, size_list, size_to_name, backbone_state_dict


def resume_svd(args, pred_state_dict):
    print('Total Tensors: {}'.format(len(pred_state_dict)))
    final_state_dict = {}
    for param_name in tqdm(pred_state_dict.keys()):
        if ('-' in param_name):
            real_name = param_name.split('-')[0].strip()
            if (real_name in final_state_dict):
                continue
            ut = pred_state_dict[f'{real_name}-ut']
            st = pred_state_dict[f'{real_name}-st'].reshape(1, 1)
            vt = pred_state_dict[f'{real_name}-vt']
            len_ut = torch.sqrt((ut * ut).sum()).item()
            len_vt = torch.sqrt((vt * vt).sum()).item()
            ut = ut / len_ut
            vt = vt / len_vt
            pred_param = ut @ st @ vt
            print(f'[Resume SVD]', param_name, ut.size(), st.size(), vt.size(), '=>', pred_param.size())
            final_state_dict[real_name] = pred_param

        else:
            assert(pred_state_dict[param_name].size(-1) == 1)
            final_state_dict[param_name] = pred_state_dict[param_name].reshape(-1)
            print(f'[No SVD]', param_name, pred_state_dict[param_name].size(), final_state_dict[param_name].size())

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_path)
    return final_state_dict, tokenizer


def resume_model(args, all_pred_ckpt, size_to_name, size_list):
    print('\n------- Resuming Delta -------')
    processed_state_dict = {}
    for sz in size_list:
        ckpt_name = '{};{}'.format(sz[0], sz[1])
        print('    --- Resuming {} ---'.format(ckpt_name))
        for idx, tensor_name in enumerate(size_to_name[ckpt_name]):
            if (sz[-1] == 1):
                pred_tensor = all_pred_ckpt[ckpt_name][:, idx:idx+1]
            else:
                pred_tensor = all_pred_ckpt[ckpt_name][idx:idx+1, :]
            assert(pred_tensor.size(0) == sz[0] and pred_tensor.size(-1) == sz[-1])
            processed_state_dict[tensor_name] = pred_tensor

    delta_state_dict, tokenizer = resume_svd(args, processed_state_dict)
    return delta_state_dict, tokenizer


def main(args):
    # Loading Training CKPTs
    train_ckpt_delta, train_ckpt_last_delta, size_list, size_to_name, real_backbone_state_dict = load_train_ckpt(args)

    last_ckpt_delta = train_ckpt_delta[-1]
    last_ckpt_last_delta = train_ckpt_last_delta[-1]

    print('\n\n====== Predicting ======')
    all_pred_ckpt = {}
    for sz in size_list:
        ckpt_name = '{};{}'.format(sz[0], sz[1])
        print('\n------ Processing {} ------'.format(sz))

        if (sz[-1] == 1):
            dim = sz[0]
            is_trans = True
        else:
            dim = sz[1]
            is_trans = False

        # Construct Models
        tgt_path = os.path.join(args.ckpt_path, f'{sz[0]}-{sz[1]}-epoch_{args.epoch}.pt')
        print('Loading checkpoints from {}'.format(tgt_path))
        param = torch.load(tgt_path)
        model = MatrixPredict(dim, loss_func=args.loss_func)
        model.load_state_dict(param)
        model.eval().to('cuda')

        src_ckpt_delta = last_ckpt_delta[ckpt_name]
        src_ckpt_last_delta = last_ckpt_last_delta[ckpt_name]
        if is_trans == True:
            src_ckpt_delta = src_ckpt_delta.t()
            src_ckpt_last_delta = src_ckpt_last_delta.t()
        
        pred_ckpt = model(
            src_ckpt_delta.to('cuda'),
            src_ckpt_last_delta.to('cuda'),
        )

        if is_trans == True:
            pred_ckpt = pred_ckpt.t()
        pred_ckpt = pred_ckpt.to('cpu')

        print(src_ckpt_delta.size(), src_ckpt_last_delta.size(), pred_ckpt.size())
        all_pred_ckpt[ckpt_name] = pred_ckpt

    delta_state_dict, tokenizer = resume_model(args, all_pred_ckpt, size_to_name, size_list)

    scale_coef = args.min_coef
    while (scale_coef <= args.max_coef):
        final_state_dict = {}
        scale_coef = round(scale_coef, 1)
        print("------- Resuming with scale_coef = {} -------".format(scale_coef))
        for nam in real_backbone_state_dict.keys():
            final_state_dict[nam] = delta_state_dict[nam] * scale_coef + real_backbone_state_dict[nam]

        tgt_path = os.path.join(args.ckpt_path, 'pred_model', 'step_{}-Based{}-PredEpoch{}-PredictExtend'.format(scale_coef, args.max_steps, args.epoch))
        print('Saving predicted model to {}'.format(tgt_path))
        model = AutoModelForCausalLM.from_pretrained(args.backbone_model_path)
        model.load_state_dict(final_state_dict)
        model.eval()
        model.save_pretrained(tgt_path)
        tokenizer.save_pretrained(tgt_path)
        scale_coef = scale_coef + args.inter_coef


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_model_path", type=str)
    parser.add_argument("--default_dir", default='hf_model', type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--delta_name", type=str)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--svd_topk", type=int)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--pred_step", type=int)
    parser.add_argument("--inter_step", type=int)
    parser.add_argument("--min_coef", type=float)
    parser.add_argument("--max_coef", type=float)
    parser.add_argument("--inter_coef", type=float)
    parser.add_argument("--loss_func", type=str, default='l2')
    args = parser.parse_args()

    main(args)