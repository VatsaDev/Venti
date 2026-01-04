# dense model to MoE changes

trying to match qwen-3-2b in 1b params for OCR-only and long ctx reasoning tasks

the only known MoE variants that seem to be param equivalent to dense are:

 - [MoEUT](https://arxiv.org/abs/2405.16039)
 - [deepseek stye MoEs (also copied and scaled longcat, kimi, etc)](https://arxiv.org/pdf/2412.19437v1)

one of these is much more practiced and standard than the other, and also has [scaling laws](https://arxiv.org/html/2507.17702v2) (EL and MoE scaling factor)

there is also the issue of sparsity at small scales, where you need a minimum number of active parameters for a good model, so there is a ceiling on the number of experts

I think the solution to that is to not have that many experts, and to instead let the neurons inside experts cluster instead, similar to [Relu^2](https://arxiv.org/pdf/2402.03804), perhaps only 8 experts and a shared expert, and topk=1+shared 

planned:

 - finegrained MoE, deepseek style,
 - the copy expert from [longcat-flash](https://arxiv.org/pdf/2509.01322)
 - MTP layers only using SWA exclusively for spec-decoding, from [MiMO flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf)
 - abalate LBL (load bear loss) vs expert bias (test regular vs pid tune)
 - relu^2 squared sparsity method on large experts to produce a higher sparsity without loss penalty

The copy expert can be used as an effective layer skip, just setup router so that if the copy expert is chosen, no other expert is chosen. from here we can train the model to skip tokens N% of the time while monitoring loss, with the goal being the model pushing simpler tokens to use less layers, an easy form of dynamic compute

The 4 MTP layers will justify their own training pretty easily even with low acc rates, for a model with 16 layers, 1 MTP head being accurate about the N+1 token 6.25% of the time basically saves a forward pass through the model, as 16 MTP passes = 1 forward pass, so even if the MTP head is right 1/16th of the time its a forward pass saved. 

Same with further MTP heads, head 2 needs to be right 3% of the time to break even, head 3 needs to be right 2% of the time to break even, head 4 needs to be right 1.5% of the time to break even, beyond this is just diminishing returns though  

