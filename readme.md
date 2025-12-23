# Venti

designed to be a high-throughput architecture

Planned features:

 - Scaling configs (16M -> 1B)
 - MuP inits and scaling
 - 256K ctx, extendable to 1M (mini trained on 4096 ctx)
 - MLA or GQA, minimize the KV cache
 - ultra-sparse MoE (1%-5% active params), along with shared experts 
 - 3/4 layers use SWA at 128 ctx (from MiMO flash paper), 4th layer uses global attn 
 - DSA deepseek lightning indexer (uses 2k for 128k, 1.5% of tokens, use 64 for 4096)
 - 4 MTP layers, all using SWA, used for spec decoding
 - Fast triton/cuteDSL Kernels for all features, all 
 - proper experimental logging, configs, containerized 
 - FP8 training or int8 QAT checkpoint, maybe NVFP4 training
 - full inferencing solution VLLM style, paged attn, KV cache, based off the mini SGlang repo
 
Completed:

 - N/A







