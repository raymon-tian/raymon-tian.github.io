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

## arxiv2003

* [Gradient-Based Deep Quantization of Neural Networks through Sinusoidal Adaptive Regularization](https://arxiv.org/pdf/2003.00146.pdf)
* [Quantized Neural Network Inference with Precision Batching](https://arxiv.org/pdf/2003.00822.pdf)
* [Generative Low-bitwidth Data Free Quantization](https://arxiv.org/pdf/2003.03603.pdf)
* [Propagating Asymptotic-Estimated Gradients for Low Bitwidth Quantized Neural Networks](https://arxiv.org/pdf/2003.04296.pdf)
* [Kernel Quantization for Efficient Network Compression](https://arxiv.org/pdf/2003.05148.pdf)
* [SIMBA: A Skyrmionic In-Memory Binary Neural Network Accelerator](https://arxiv.org/pdf/2003.05132.pdf)
* 【samsung】[BATS: Binary ArchitecTure Search](https://arxiv.org/pdf/2003.01711.pdf)
* 【attention】[ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions](https://arxiv.org/pdf/2003.03488.pdf)
* 【CVPR2020】[BiDet: An Efficient Binarized Object Detector](https://arxiv.org/pdf/2003.03961.pdf)
* [Deep Molecular Programming: A Natural Implementation of Binary-Weight ReLU Neural Networks](https://arxiv.org/pdf/2003.13720.pdf)
* [A Power-Efficient Binary-Weight Spiking Neural Network Architecture for Real-Time Object Classification](https://arxiv.org/pdf/2003.06310.pdf)
* 【ICASSP2020】[LANCE: Efficient Low-Precision Quantized Winograd Convolution for Neural Networks Based on Graphics Processing Units](https://arxiv.org/pdf/2003.08646.pdf)
* [Compressing deep neural networks on FPGAs to binary and ternary precision with HLS4ML](Compressing deep neural networks on FPGAs to binary and ternary precision with HLS4ML)
* 【samsung】【ICLR2020】[Training Binary Neural Networks with Real-to-Binary Convolutions](https://arxiv.org/pdf/2003.11535.pdf)

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

## arxiv2003

* [Learned Threshold Pruning](https://arxiv.org/pdf/2003.00075.pdf)
* [Sparsity Meets Robustness: Channel Pruning for the Feynman-Kac Formalism Principled Robust Deep Neural Nets](https://arxiv.org/pdf/2003.00631.pdf)
* [Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection](https://arxiv.org/pdf/2003.01794.pdf)
* [Privacy-preserving Learning via Deep Net Pruning](https://arxiv.org/pdf/2003.01876.pdf)
* 【ICLR2020】[Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://arxiv.org/pdf/2003.02389.pdf)
* [Cluster Pruning: An Efficient Filter Pruning Method for Edge AI Vision Applications](https://arxiv.org/pdf/2003.02449.pdf)
* [Pruning Filters while Training for Efficiently Optimizing Deep Learning Networks](https://arxiv.org/pdf/2003.02800.pdf)
* 【MLSys2020】[What is the State of Neural Network Pruning?](https://arxiv.org/pdf/2003.03033.pdf)
* [Morfessor EM+Prune: Improved Subword Segmentation with Expectation Maximization and Pruning](https://arxiv.org/pdf/2003.03131.pdf)
* 【intel】[Channel Pruning via Optimal Thresholding](https://arxiv.org/pdf/2003.04566.pdf)
* [AP-MTL: Attention Pruned Multi-task Learning Model for Real-time Instrument Detection and Segmentation in Robot-assisted Surgery](https://arxiv.org/pdf/2003.04769.pdf)
* [SASL: Saliency-Adaptive Sparsity Learning for Neural Network Acceleration](https://arxiv.org/pdf/2003.05891.pdf)
* [DA-NAS: Data Adapted Pruning for Efficient Neural Architecture Search](https://arxiv.org/pdf/2003.12563.pdf)
* [How Not to Give a FLOP: Combining Regularization and Pruning for Efficient Inference](https://arxiv.org/pdf/2003.13593.pdf)
* [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/pdf/2003.13683.pdf)
* [Improved Gradient based Adversarial Attacks for Quantized Networks](https://arxiv.org/pdf/2003.13511.pdf)
* [A Privacy-Preserving DNN Pruning and Mobile Acceleration Framework](https://arxiv.org/pdf/2003.06513.pdf)
* 【AAAI2020】[Channel Pruning Guided by Classification Loss and Feature Importance](https://arxiv.org/pdf/2003.06757.pdf)
* 【CVPR2020】[Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/pdf/2003.08935.pdf)
* [SPFCN: Select and Prune the Fully Convolutional Networks for Real-time Parking Slot Detection](https://arxiv.org/pdf/2003.11337.pdf)
* [Event-based Asynchronous Sparse Convolutional Networks](Event-based Asynchronous Sparse Convolutional Networks)
* 【ICLR2020-workshop】[Data Parallelism in Training Sparse Neural Networks](https://arxiv.org/pdf/2003.11316.pdf)
* [MINT: Deep Network Compression via Mutual Information-based Neuron Trimming](MINT: Deep Network Compression via Mutual Information-based Neuron Trimming)
* [DP-Net: Dynamic Programming Guided Deep Neural Network Compression](https://arxiv.org/pdf/2003.09615.pdf)

# distillation

## arxiv2001

* [Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification](https://arxiv.org/pdf/2001.01536.pdf)
* [Hydra: Preserving Ensemble Diversity for Model Distillation](https://arxiv.org/pdf/2001.04694.pdf)

## arxiv2002

* [Understanding and Improving Knowledge Distillation](https://arxiv.org/pdf/2002.03532.pdf)
* 【NLP】[TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval](https://arxiv.org/pdf/2002.06275.pdf)
* [Residual Knowledge Distillation](https://arxiv.org/pdf/2002.09168.pdf)
* [An Efficient Method of Training Small Models for Regression Problems with Knowledge Distillation](An Efficient Method of Training Small Models for Regression Problems with Knowledge Distillation)

## arxiv2003

* [On the Unreasonable Effectiveness of Knowledge Distillation: Analysis in the Kernel Regime](https://arxiv.org/pdf/2003.13438.pdf)
* [Squeezed Deep 6DoF Object Detection Using Knowledge Distillation](https://arxiv.org/pdf/2003.13586.pdf)
* [Spatio-Temporal Graph for Video Captioning with Knowledge Distillation](Spatio-Temporal Graph for Video Captioning with Knowledge Distillation)
* 【CVPR2020】[Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model](https://arxiv.org/pdf/2003.13960.pdf)
* 【CVPR2020】[Regularizing Class-wise Predictions via Self-knowledge Distillation](Regularizing Class-wise Predictions via Self-knowledge Distillation)

# NAS

## arxiv2002

* [Learning Architectures for Binary Networks](https://arxiv.org/pdf/2002.06963.pdf)

## arxiv2003

* [PONAS: Progressive One-shot Neural Architecture Search for Very Efficient Deployment](https://arxiv.org/pdf/2003.05112.pdf)
* [Accelerator-aware Neural Network Design using AutoML](https://arxiv.org/pdf/2003.02838.pdf)
* 【Attention】**[Efficient Bitwidth Search for Practical Mixed Precision Neural Network](Efficient Bitwidth Search for Practical Mixed Precision Neural Network)**

# others

* 【MLSys2020-workshop】[Efficient Memory Management for Deep Neural Net Inference](Efficient Memory Management for Deep Neural Net Inference)
* [DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-chip Training](https://arxiv.org/pdf/2003.06471.pdf)
* [TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks](https://arxiv.org/pdf/2003.09855.pdf)
* 【CVPR2020】[Resolution Adaptive Networks for Efficient Inference](Resolution Adaptive Networks for Efficient Inference)

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
* [Compact recurrent neural networks for acoustic event detection on low-energy low-complexity platforms](https://arxiv.org/pdf/2001.10876.pdf)

# Training Acceleration

* [STANNIS: Low-Power Acceleration of Deep Neural Network Training Using Computational Storage](https://arxiv.org/pdf/2002.07215.pdf)
* 【Google】[Low-rank Gradient Approximation For Memory-Efficient On-device Training of Deep Neural Network](https://arxiv.org/pdf/2001.08885.pdf)