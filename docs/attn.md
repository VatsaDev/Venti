# Attn changes

`INCOMPLETE`

planned features:

 - KV sharing across global layers (from [MiMO flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf)) 
 - MQA (from [one write head all you need](https://arxiv.org/pdf/1911.02150))
 - qwen-size ctx window (256k, extendable to 1M), most likely 128k or less during pre-train, midtrain yarn scale
 - fp8 (fully supported on hopper and blackwell, halves bandwidth and KV, doubles flops)
 - changing MQA into SSA by compressing the 1 k and 1 v into a KV latent fr every layer
 - deepseek lightning indexer (possibly run it on a SWA layer for performance)
 - MTP heads using SWA (from [MiMO flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf))

completed:

 - 128 ctx SWA window (from [MiMO flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf)) (still needs flex_attn)
 - hybrid ratios (from [noam shazeer char.ai research](https://web.archive.org/web/20250325023214/https://research.character.ai/optimizing-inference/))

using flash attn mask, 

global/local layers with SWA, using 3:1 ratio, or 3/4 layers are SWA, window size = 128

doesnt work well on the T4, need to upgrade to h100+flex_attn

KV cache:

in an 8 head MHA, 16 layers, fp16, 4096 ctx: `256 mb`
in an 8 head MHA, 5 global layers, 12 local layers fp16, 4096 ctx: `80mb + 6mb, 86mb`

its already a 67% reduction

with attn_sinks this should match all global




