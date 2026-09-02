# Ouroboros Triton

![tests](https://github.com/vivek5200/ouroboros-triton/actions/workflows/tests.yml/badge.svg)

C++/Triton bare-metal engine for the Ouroboros v7.1 code refactoring system.

## Modules
- **Module 2 (Block Memory)**: 64-token fixed-size physical blocks via C++ array-backed linked list
- **Module 6 (Serving)**: Supervisor-Worker HA layer with stateless restartable workers

## Constraints
- NEVER use PyTorch tensor reshapes for dynamic sequence expansion
- On Grace Blackwell sm_121a: wrap CUDA graph capture with TLB warmup
- Workers are stateless; Supervisor owns the KV-cache and block table
