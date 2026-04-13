# NExt: Nonlinear Extrapolation of Low-rank Optimization Trajectories

## Introduction
The official repository of "Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration".

To mitigate the substantial computational cost introduced by scaling RLVR for LLMs, we aim to predict the model’s future states based on historical optimization trajectories, thereby reducing the number of RLVR training steps and improving training efficiency.

## Quick Start

1. Convert the LoRA trained checkpoints into the Hugging Face model format
```
python src/merge_lora.py
```


2. Compute `Global Delta`, `Local Delta`, and `Target Delta`. Since this part of the code does not involve GPU computation, multiple threads can be used to accelerate the decomposition process.

```
python src/compute_global_delta.sh
python src/compute_local_delta.sh
python src/compute_target_delta.sh
```

3. Train the predictor to model the optimization trajectory.
```
bash scripts/run_train.sh
```

4. Extrapolate the LLM paramters based on the trained predictor.
```
bash scripts/run_extrapolate.sh
```

## Citation
Please kindly cite our reports if they are helpful for your research.

```
@article{Chen2026NExt,
  title={Low-rank Optimization Trajectories Modeling for LLM RLVR Acceleration},
  author={Chen, Zhipeng and Qian, Tao and Zhao, Wayne Xin and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:},
  year={2026}
}
```