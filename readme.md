# Venti

trained a 15M under the "baseline" commit, revert back to here for any baseline testing

 - loss, also bits per byte
 - perplexity
 - flops used 
 - BPB on zero shot hellaswag (looking for any upward rise) <br><br> <img width="752" height="124" alt="image" src="https://github.com/user-attachments/assets/757e2d87-80bb-4a3d-9977-ba8c3475bed3" />

## details

designed to be a high-throughput architecture

Planned features:

 - 256K ctx, extendable to 1M (mini trained on 4096 ctx)
 - MLA or GQA, minimize the KV cache
 - ultra-sparse MoE (1%-5% active params), along with shared experts  
 - DSA deepseek lightning indexer (uses 2k for 128k, 1.5% of tokens, use 64 for 4096)
 - 4 MTP layers, all using SWA, used for spec decoding
 - Fast triton/cuteDSL Kernels for big features (high mfu) 
 - FP8 training (hopper)
 - add model to VLLM or SGlang internally, (ex. inside `model_executors` for vllm, use the paged_attn in forward pass)
 
Completed:

 - scaling configs (0.08M test, 15M small, 1B big)
 - MuP rules
 - HP sweeps 
 - 3/4 layers use SWA at 128 ctx (from MiMO flash paper)
 - proper experimental logging, configs, *containerized (need to drop in docker later) 
