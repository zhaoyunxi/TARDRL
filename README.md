# TARDRL: Task-Aware Reconstruction for Dynamic Representation Learning of fMRI
## TARDRL
Task-Aware Reconstruction Dynamic Rpresentation Learning (TARDRL) is a novel algorithm to improve prediction performance with task-aware reconstruction. Different from the conventional sequential design on the reconstruction and prediction, we set up the reconstruction and prediction task in a parallel paradigm to learn robust task-aware representations (training TARDRL in a multi-task learning way).
<p align="center">
<img src=assets/idea_img.png />
</p>

Based on the parallelized framework, we adopt an attention-guided masking strategy to mask ROIs important to the prediction task. Specifically, it uses the attention maps generated by STF during prediction and determines the set of ROIs to mask. During reconstruction, signals of these ROIs are masked out and reconstructed, facilitating the reconstruction in a task-aware manner. 
<p align="center">
<img src=assets/idea_img1.png />
</p>
