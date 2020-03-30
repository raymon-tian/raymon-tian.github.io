---
layout:     post
title:      模型压缩与加速论文列表
subtitle:   剪枝·量化·蒸馏等
date:       2020-03-25
author:     DT
header-img: img/post-bg-debug.png
catalog: true
tags:
    - model.acceleration
    - paper.list


---

> 模型压缩与加速论文列表

# quantization

## arxiv2001

* 【CVPR2020】[ZeroQ: A Novel Zero Shot Quantization Framework](ZeroQ: A Novel Zero Shot Quantization Framework)
* 【Apple】[Least squares binary quantization of neural networks](https://arxiv.org/pdf/2001.02786.pdf)
* [RPR: Random Partition Relaxation for Training; Binary and Ternary Weight Neural Networks](https://arxiv.org/pdf/2001.01091.pdf)
* 【ICLR2020】[Shifted and Squeezed 8-bit Floating Point format for Low-Precision Training of Deep Neural Networks](https://arxiv.org/pdf/2001.05674.pdf)
* 【Alibaba】[MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?](https://arxiv.org/pdf/2001.05936.pdf)

## arxiv2002

* [Automatic Pruning for Quantized Neural Networks](https://arxiv.org/pdf/2002.00523.pdf)
* [BitPruning: Learning Bitlengths for Aggressive and Accurate Quantization](https://arxiv.org/pdf/2002.03090.pdf)
* [Post-Training Piecewise Linear Quantization for Deep Neural Networks](https://arxiv.org/pdf/2002.00104.pdf)
* [SQWA: Stochastic Quantized Weight Averaging for Improving the Generalization Capability of Low-Precision Deep Neural Networks](https://arxiv.org/pdf/2002.00343.pdf)
* [Widening and Squeezing: Towards Accurate and Efficient QNNs](https://arxiv.org/pdf/2002.00555.pdf)
* [Towards Explainable Bit Error Tolerance of Resistive RAM-Based Binarized Neural Networks](https://arxiv.org/pdf/2002.00909.pdf)
* [Switchable Precision Neural Networks](https://arxiv.org/pdf/2002.02815.pdf)
* [Robust Quantization: One Model to Rule Them All](https://arxiv.org/pdf/2002.07686.pdf)
* [SYMOG: learning symmetric mixture of Gaussian modes for improved fixed-point quantization](https://arxiv.org/pdf/2002.08204.pdf)
* [Post-training Quantization with Multiple Points: Mixed Precision without Mixed Precision](https://arxiv.org/pdf/2002.09049.pdf)
* [Searching for Winograd-aware Quantized Networks](https://arxiv.org/pdf/2002.10711.pdf)
* [Small-Footprint Open-Vocabulary Keyword Spotting with Quantized LSTM Networks](https://arxiv.org/pdf/2002.10851.pdf)
* [Optimal Gradient Quantization Condition for Communication-Efficient Distributed Training](https://arxiv.org/pdf/2002.11082.pdf)
* 【MLSys2020】[PoET-BiN: Power Efficient Tiny Binary Neurons](https://arxiv.org/pdf/2002.09794.pdf)
* [sBSNN: Stochastic-Bits Enabled Binary Spiking Neural Network with On-Chip Learning for Energy Efficient Neuromorphic Computing at the Edge](https://arxiv.org/pdf/2002.11163.pdf)
* 【intel】【Framework】[**Neural Network Compression Framework for fast model inference**](https://arxiv.org/pdf/2002.08679.pdf)
* 【ICLR2020】[Precision Gating: Improving Neural Network Efficiency with Dynamic Dual-Precision Activations](https://arxiv.org/pdf/2002.07136.pdf)
* 【ICLR2020】[BinaryDuo: Reducing Gradient Mismatch in Binary Activation Network by Coupling Binary Activations](https://arxiv.org/pdf/2002.06517.pdf)
* [Learning Architectures for Binary Networks](https://arxiv.org/pdf/2002.06963.pdf)
* [Exploring the Connection Between Binary and Spiking Neural Networks](Exploring the Connection Between Binary and Spiking Neural Networks)
* [Training Binary Neural Networks using the Bayesian Learning Rule](https://arxiv.org/pdf/2002.10778.pdf)

# pruning

## arxiv2001

* 【ASPLOS2020-CCF-A】[PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning](https://arxiv.org/pdf/2001.00138.pdf)
* 【Extended version of the NeurIPS2018】[Discrimination-aware Network Pruning for Deep Model Compression](https://arxiv.org/pdf/2001.01050.pdf)
* [Learning fine-grained search space pruning and heuristics for combinatorial optimization](https://arxiv.org/pdf/2001.01230.pdf)
* [Investigation and Analysis of Hyper and Hypo neuron pruning to selectively update neurons during Unsupervised Adaptation](https://arxiv.org/pdf/2001.01755.pdf)
* [Pruning Convolutional Neural Networks with Self-Supervision](https://arxiv.org/pdf/2001.03554.pdf)
* [A “Network Pruning Network” Approach to Deep Model Compression](https://arxiv.org/pdf/2001.05545.pdf)
* 【FB】[On Iterative Neural Network Pruning, Reinitialization, and the Similarity of Masks](https://arxiv.org/pdf/2001.05050.pdf)
* 【CPAIOR2020-CCF-B】[Lossless Compression of Deep Neural Networks](https://arxiv.org/pdf/2001.00218.pdf)

## arxiv2002

* [Automatic Pruning for Quantized Neural Networks](https://arxiv.org/pdf/2002.00523.pdf)
* [Proving the Lottery Ticket Hypothesis: Pruning is All You Need](https://arxiv.org/pdf/2002.00585.pdf)
* [Activation Density driven Energy-Efficient Pruning in Training](https://arxiv.org/pdf/2002.02949.pdf)
* [Convolutional Neural Network Pruning Using Filter Attenuation](https://arxiv.org/pdf/2002.03299.pdf)
* [Lookahead: A Far-Sighted Alternative of Magnitude-based Pruning](https://arxiv.org/pdf/2002.04809.pdf)
* [PCNN: Pattern-based Fine-Grained Regular Pruning towards Optimizing CNN Accelerators](https://arxiv.org/pdf/2002.04997.pdf)
* [Layer-wise Pruning and Auto-tuning of Layer-wise Learning Rates in Fine-tuning of Deep Networks](https://arxiv.org/pdf/2002.06048.pdf)
* [Training Efficient Network Architecture and Weights via Direct Sparsity Control](https://arxiv.org/pdf/2002.04301.pdf)
* [Knapsack Pruning with Inner Distillation](https://arxiv.org/pdf/2002.08258.pdf)
* [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/pdf/2002.08307.pdf)
* [Performance Aware Convolutional Neural Network Channel Pruning for Embedded GPUs](https://arxiv.org/pdf/2002.08697.pdf)
* [Gradual Channel Pruning while Training using Feature Relevance Scores for Convolutional Neural Networks](https://arxiv.org/pdf/2002.09958.pdf)
* [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/pdf/2002.10179.pdf)
* [On Pruning Adversarially Robust Neural Networks](https://arxiv.org/pdf/2002.10509.pdf)
* [Calibrate and Prune: Improving Reliability of Lottery Tickets Through Prediction Calibration](https://arxiv.org/pdf/2002.03875.pdf)
* 【Oxford】[Pruning untrained neural networks: Principles and Analysis](https://arxiv.org/pdf/2002.08797.pdf)
* 【MNN,MLSys2020】[MNN: A Universal and Efficient Inference Engine](https://arxiv.org/pdf/2002.12418.pdf)

# distillation

## arxiv2001

* [Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification](https://arxiv.org/pdf/2001.01536.pdf)
* [Hydra: Preserving Ensemble Diversity for Model Distillation](https://arxiv.org/pdf/2001.04694.pdf)

## arxiv2002

* [Understanding and Improving Knowledge Distillation](https://arxiv.org/pdf/2002.03532.pdf)
* 【NLP】[TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval](https://arxiv.org/pdf/2002.06275.pdf)
* [Residual Knowledge Distillation](https://arxiv.org/pdf/2002.09168.pdf)
* [An Efficient Method of Training Small Models for Regression Problems with Knowledge Distillation](An Efficient Method of Training Small Models for Regression Problems with Knowledge Distillation)

# NAS

## arxiv2002

* [Learning Architectures for Binary Networks](https://arxiv.org/pdf/2002.06963.pdf)

# others

## arxiv2001

* 【MLSys2020-workshop】[Efficient Memory Management for Deep Neural Net Inference](Efficient Memory Management for Deep Neural Net Inference)

## arxiv2002

* 【AAAI2020】[DWM: A Decomposable Winograd Method for Convolution Acceleration](https://arxiv.org/pdf/2002.00552.pdf)
* [Accelerating Deep Learning Inference via Freezing](https://arxiv.org/pdf/2002.02645.pdf)
* 【HPCA2020】[SpArch: Efficient Architecture for Sparse Matrix Multiplication](https://arxiv.org/pdf/2002.08947.pdf)
* [A Systematic Survey of General Sparse Matrix-Matrix Multiplication](https://arxiv.org/pdf/2002.11273.pdf)
* [Communication-Efficient Edge AI: Algorithms and Systems](https://arxiv.org/pdf/2002.09668.pdf)
* [Balancing Efficiency and Flexibility for DNN Acceleration via Temporal GPU-Systolic Array Integration](https://arxiv.org/pdf/2002.08326.pdf)
* 【DATE2020】[TFApprox: Towards a Fast Emulation of DNN Approximate Hardware Accelerators on GPU](https://arxiv.org/pdf/2002.09481.pdf)
* 【IC2E2020】[MDInference: Balancing Inference Accuracy and Latency for Mobile Applications](MDInference: Balancing Inference Accuracy and Latency for Mobile Applications)
* [HOTCAKE: Higher Order Tucker Articulated Kernels for Deeper CNN Compression](https://arxiv.org/pdf/2002.12663.pdf)



# NLP

* [Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers)
* [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://arxiv.org/pdf/2002.11985.pdf)
* [Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation](https://arxiv.org/pdf/2002.10345.pdf)
* [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/pdf/2002.10957.pdf)
* 【Framework】[TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](https://arxiv.org/pdf/2002.12620.pdf)
* 【Alibaba】[AdaBERT: Task-Adaptive BERT Compression with Differentiable Neural Architecture Search](https://arxiv.org/pdf/2001.04246.pdf)





# Training Acceleration

* [STANNIS: Low-Power Acceleration of Deep Neural Network Training Using Computational Storage](https://arxiv.org/pdf/2002.07215.pdf)