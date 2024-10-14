# TARDRL: Task-Aware Reconstruction for Dynamic Representation Learning of fMRI
## TARDRL
**Task-Aware Reconstruction Dynamic Rpresentation Learning** (TARDRL) is a novel algorithm to improve prediction performance with task-aware reconstruction. Different from the conventional sequential design on the reconstruction and prediction, we set up the reconstruction and prediction task in a parallel paradigm to learn robust task-aware representations (training TARDRL in a multi-task learning way).
<p align="center">
<img src=assets/idea_img.png />
</p>

Based on the parallelized framework, we adopt an attention-guided masking strategy to mask ROIs important to the prediction task. Specifically, it uses the attention maps generated by STF during prediction and determines the set of ROIs to mask. During reconstruction, signals of these ROIs are masked out and reconstructed, facilitating the reconstruction in a task-aware manner. 
<p align="center">
<img src=assets/idea_img1.png width=400 heigh=150/>
</p>

## Overview
<p align="center">
<img src=assets/model_big.png />
</p>

**TARDRL** comprises four major components: an attention-guided mask layer only activated during reconstruction, a shared encoder composed of spatial transformer (STF) and temporal transformer (TTF), a predictor for prediction tasks, and a decoder for task-aware reconstruction.

## Environment setup
Create and activate conda environment named ```tardrl``` from our ```environment.yaml```
```sh
conda env create -f environment.yaml
conda activate tardrl
```
## Prepare Data

## Run

## Citation
```
@inproceedings{zhao2024tardrl,
  title={TARDRL: Task-Aware Reconstruction for Dynamic Representation Learning of fMRI},
  author={Zhao, Yunxi and Nie, Dong and Chen, Geng and Wu, Xia and Zhang, Daoqiang and Wen, Xuyun},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={700--710},
  year={2024},
  organization={Springer}
}

```
