# From Prediction to Mastery: Reality-Aligned Cognitive Diagnosis Networks
Yupei Zhang, Shuyu Yan
## Overview
> Most neural network-based cognitive diagnosis (CD) models pursue high accuracy in predicting student performance. However, CD models are often employed to figure out the mastery status of knowledge concepts (KC) in exam evaluations. To align CD networks with mastery understanding, this paper introduces RACD, i.e., Reality-Aligned Cognitive Diagnosis networks, which
aims to yield the explainable results on student-KC matrix. More specifically, RACD presents three novelties: 1) A self-learning Qmatrix that adaptively refines the alignment between questions and their associated KCs, enhancing robustness to the empirical Q-matrix; 2) A hierarchical mastery constraint that incorporates tree-structured dependencies among KCs, enforcing the learned
mastery matrix to be consistent with knowledge structures; 3) A binarization-oriented constraint on student–KC mastery degree, which drives the KC proficiency toward 0/1 distributions, thereby better reflecting the binary nature of knowledge acquisition. The proposed RACD is implemented by using the neural-network CD framework. Experimental results on multiple benchmark datasets
show that RACD results in the more practical student-KC matrix with high mastery interpretations, while maintaining competitive predictive accuracy, than the state-of-the-art methods.

## Dependencies
>The code requires Python >= 3.8 and PyTorch >= 1.10.1. CUDA 12.6

## Run
>python train.py {device} {epoch} {alpha_bin} {alpha_Q} {alpha_hier}

## Contact
>Please contact to ypzhaang@nwpu.edu.cn for any problems.
