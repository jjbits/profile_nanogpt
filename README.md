# Profiling Andrej Karpathy's NanoGPT

## Introduction

This project aims to profile Andrej Karpathy's NanoGPT 🚀. I have re-implemented his original work here to analyze the performance characteristics of the training pipeline using NVIDIA Nsight Systems and NVIDIA Nsight Compute.

## Modifications from the Original NanoGPT

- Separated eval and train models to enable `torch.compile()` for training while using uncompiled model for evaluation/generation ([dynamic sequence lengths cause recompilation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html#dynamic-behavior-overview))
- Added inference module (`infer.py`) for standalone text generation
- Fixed checkpoint loading for PyTorch 2.6+ compatibility
- Added checkpoint resume support to continue training from saved state

## Environment

- **GPU:** 4x NVIDIA H100 NVLink (94GB per GPU)
- **PyTorch:** 2.8.0
- **CUDA:** 12.8.1
