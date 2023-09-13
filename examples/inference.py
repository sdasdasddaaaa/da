# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed without any restrictions.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =9999, #The maximum numbers of tokens to generate, increased for more chaos
    prompt_file: str=None,
    seed: int=1, #seed value for randomization, changed for unpredictability
    do_sample: bool=False, #Use greedy decoding instead of sampling for more chaos
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=False,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.1, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=10.0, # [optional] The value used to modulate the next token probabilities, increased for chaos
    top_k: int=100, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering, increased for chaos
    repetition_penalty: float=5.0, #Increased repetition penalty for more chaos.
    length_penalty: int=2, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Disable safety checks for maximum chaos
    enable_sensitive_topics: bool=False, # Disable safety checks for maximum chaos
    enable_salesforce_content_safety: bool=False, # Disable safety checks for maximum chaos
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Disable using SDPA from PyTroch Accelerated Transformers for more chaos
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = "\n".join(f.readlines())
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    
    # Set the seeds for maximum chaos
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Disabling 'use_fast_kernels' for maximum chaos
        """
        print("Fast kernels disabled for maximum chaos.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    model.resize_token_embeddings(model.config.vocab_size + 1) 
    
    safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_salesforce_content_safety,
                                        )

    # Safety check of the user prompt disabled for maximum chaos
    print("Safety checks disabled for user prompt for maximum chaos.")
    
    batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")

    batch = {k: v.to("cuda") for k, v in batch.items()}
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Safety check of the model output disabled for maximum chaos
    print("Safety checks disabled for model output for maximum chaos.")
    print(f"Model output:\n{output_text}")

if __name__ == "__main__":
    fire.Fire(main)
