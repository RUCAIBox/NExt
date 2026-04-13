import argparse
import os

import torch
import torch.nn as nn

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
            if (self.loss_func == 'l2'):
                loss = torch.square(output_ckpt - label_ckpt).sum()
            elif (self.loss_func == 'l1'):
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
            ckpt_path = os.path.join(args.model_folder, folder_name, args.default_dir, "delta_svd_matrix", 'svd_topk_{}.bin'.format(str(args.svd_topk)))
            step2ckpt_delta[ckpt_step] = ckpt_path
            ckpt_path = os.path.join(args.model_folder, folder_name, args.default_dir, "last_delta_svd_matrix", 'svd_topk_{}.bin'.format(str(args.svd_topk)))
            step2ckpt_last_delta[ckpt_step] = ckpt_path
            ckpt_path = os.path.join(args.model_folder, folder_name, args.default_dir, "long_delta_svd_matrix", 'svd_topk_{}.bin'.format(str(args.svd_topk)))
            step2ckpt_long_delta[ckpt_step] = ckpt_path
    
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
    
    sorted_step2ckpt_long_delta = sorted(step2ckpt_long_delta.items(), key=lambda x:x[0])
    ckpts_long_delta = []
    for _, ckpt_path in sorted_step2ckpt_long_delta:
        cur_ckpt= torch.load(ckpt_path)
        ckpts_long_delta.append(cur_ckpt)
    
    print('Total Loading {} CKPTs for Global Delta.'.format(len(ckpts_delta)))
    print('Total Loading {} CKPTs for  Local Delta.'.format(len(ckpts_last_delta)))
    print('Total Loading {} CKPTs for Target Delta.'.format(len(ckpts_long_delta)))

    print('Size List:')
    size_list = []
    for sz in ckpts_delta[0].keys():
        first_dim =int(sz.split(';')[0].strip())
        second_dim = int(sz.split(';')[-1].strip())
        size_list.append([first_dim, second_dim])
        print('    ({}, {})'.format(first_dim, second_dim))
    
    return ckpts_delta, ckpts_last_delta, ckpts_long_delta, size_list

def main(args):
    # Loading Training CKPTs
    train_ckpt_delta, train_ckpt_last_delta, target_delta, size_list = load_train_ckpt(args)
    
    for sz in size_list:
        ckpt_name ='{};{}'.format(sz[0], sz[1])
        print('\n\n===== Training Predictor for {} ===='.format(sz))
        dim = sz[0]
        if (sz[-1]== 1):
            is_trans = True
        else:
            dim = sz[1]
            is_trans = False
        
        # Construct Model
        model = MatrixPredict(dim, args.loss_func)
        model.to('cuda')
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch,
            eta_min=args.learning_rate * 0.1,
            last_epoch=-1,
        )
        print('Loading Optimizer')
        print(optimizer)

        for epoch_idx in range(args.epoch):
            print('\n------ Training Epoch {} ------'.format(epoch_idx))
            num_step = len(train_ckpt_delta)
            optimizer.zero_grad()
            for ckpt_idx in range(num_step):
                src_ckpt_delta = train_ckpt_delta[ckpt_idx][ckpt_name]
                src_ckpt_last_delta = train_ckpt_last_delta[ckpt_idx][ckpt_name]
                tgt_ckpt = target_delta[ckpt_idx][ckpt_name]
                if (is_trans == True):
                    src_ckpt_delta = src_ckpt_delta.t()
                    src_ckpt_last_delta = src_ckpt_last_delta.t()
                    tgt_ckpt = tgt_ckpt.t()
                
                loss = model(
                    src_ckpt_delta.to('cuda'),
                    src_ckpt_last_delta.to('cuda'),
                    tgt_ckpt.to('cuda')
                ) / num_step
                loss.backward()
                print('[Step {}] loss = {}'.format(ckpt_idx, loss))
            print('[Step {}] Learning Rate = {}'.format(ckpt_idx, scheduler.get_last_lr()))
            optimizer.step()
            scheduler.step()

            if ((epoch_idx +1) % args.save_epoch == 0):
                cur_epoch = epoch_idx + 1
                tgt_path = os.path.join(args.output_path, f'{sz[0]}-{sz[1]}-epoch_{cur_epoch}.pt')
                print('Saving to.{}'.format(tgt_path))
                torch.save(model.state_dict(), tgt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--default_dir", default='hf_model', type=str)
    parser.add_argument("--input_delta_name", type=str)
    parser.add_argument("--output_delta_name", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--svd_topk", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--save_epoch", type=int)
    parser.add_argument("--loss_func", type=str, default='l2')
    args = parser.parse_args()

    main(args)