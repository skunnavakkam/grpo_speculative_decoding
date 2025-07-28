# Speculative Decoding for GRPO

GRPO is famously inference bound. What if we tried to get around it by using speculative decoding? (a google search for `grpo speculative decoding` yields no results, so maybe this is new)

I'm going to use $M_L$ with parameters $\theta_L$ to denote the large model, and $M_S$ with parameters $\theta_S$ to denote the speculative model. One of the issues that I imagine happens with GRPO is that the large model you use ends up shifting distribution from $\theta_L$ to $\theta_L^*$ which significantly worsens the quality of the speculative model.

What I want to try is to use the large model to SFT / CPT the speculative model every $n_{update}$ steps such that the divergence between $M_L(\theta_L*)$ and $M_S(\theta_S*)$ is minimized.

Let's start by doing this with vLLM! Since vLLM / tgi / sgLang (all transformer inference libraries) only support using medusa / eagle as draft models, we instead shall continously fine-tune medusa. 


