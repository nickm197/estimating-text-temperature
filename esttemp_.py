"""
Estimating Text Temperature with Language Models

Companion code for the paper: "Estimating Text Temperature with Language Models"
This script evaluates the temperature of texts by calculating 
the Maximum Likelihood Estimate (MLE) of the temperature parameter (T) for a given text.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from scipy.optimize import root_scalar
import torch.nn.functional as F
import os
import glob
import gc
from collections import defaultdict
import json
from tqdm import tqdm

# Length of the prompt used to generate the text. 
# Set to 0 because the analyzed texts were generated from a random single token.
# Adjust this if estimating the temperature of a completion where the prompt 
# should be excluded from the MLE calculation.
promptlen = 0 

# Maximum character length to process to prevent Out-Of-Memory (OOM) errors
maxlen = 20000

def estimate_temperature_fast(logits, selected_indices):
    """
    Estimates temperature using Root Finding on the gradient of the Log-Likelihood.
    Performs heavy matrix operations on the GPU (if available).
    
    This function numerically solves Equation 5 from the paper:
    Sum(u_obs) = Sum(E[u|T])
    
    Instead of solving directly for T, it solves for beta = 1/T for better numerical stability.

    Args:
        logits (torch.Tensor): Shape (N, V). On GPU. Unscaled logits from the model.
        selected_indices (torch.Tensor): Shape (N,). On GPU. The actual tokens observed in the text.
        
    Returns:
        float: The estimated temperature T = 1/beta, or np.nan if convergence fails.
    """
    # Ensure inputs are tensors
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(selected_indices, torch.Tensor):
        selected_indices = torch.tensor(selected_indices)

    # 1. Pre-calculate the sum of observed logits (This is constant w.r.t Beta)
    # We use gather to pick specific indices: logits[i, selected_indices[i]]
    # This represents the left side of the MLE Equation: \sum u_obs
    u_obs_sum = logits.gather(1, selected_indices.unsqueeze(1)).sum().item()

    # 2. Define the gradient function: G(beta) = Sum(u_obs) - Sum(E[u|beta])
    def gradient_function(beta):
        # Handle edge case where solver tries beta <= 0 (unphysical, T must be > 0)
        if beta <= 1e-5: beta = 1e-5

        # Scale logits by beta (inverse temp)
        scaled_logits = logits * beta

        # Use PyTorch's optimized Softmax (fused kernel)
        # It automatically handles numerical stability (subtracting max)
        probs = F.softmax(scaled_logits, dim=1)

        # Calculate Expected Value E[u] = Sum(u * p(u))
        # Element-wise multiplication followed by sum over vocabulary
        expected_logits = torch.sum(probs * logits, dim=1)

        # The gradient is Observed_Sum - Expected_Sum
        # We want to find the root where this difference is 0
        grad = u_obs_sum - expected_logits.sum().item()
        return grad

    # 3. Solve for G(beta) = 0 using scipy's root_scalar (Brent's method by default)
    try:
        sol = root_scalar(gradient_function, bracket=[0.01, 10000.0], xtol=1e-8, x0=5000 )
    except ValueError:
        print('Failure')
        return np.nan
    else:
        optimal_beta = sol.root
        return 1.0 / optimal_beta

def compute_logits_from_text(model, tokenizer, text):
    """
    Takes an existing text string, tokenizes it, and runs a forward pass
    to calculate the logits for every token in the sequence.
    """
    # 1. Tokenize the text
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids[0] # Flatten batch dimension

    # 2. Forward Pass (No generation, just evaluation of existing text)
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        all_logits = outputs.logits[0] # Shape: (Seq_Len, Vocab)

    # 3. Alignment (The "Next Token Prediction" shift)
    # The model outputs logits at position `t` predicting the token at `t+1`.
    # We want to match logits[t] with input_ids[t+1].

    # We remove the last logit (it predicts the token *after* our text ends)
    logits_aligned = all_logits[promptlen:-1, :]

    # We remove the first input_id (it has no preceding logit predicting it)
    tokens_aligned = input_ids[promptlen+1:]

#    return logits_aligned.cpu().float().numpy(), tokens_aligned.cpu().numpy()
    return logits_aligned, tokens_aligned

# List of models evaluated in the paper, covering different families, 
# sizes, and architectures (Base/Instruct versions).
model_ids = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-Base",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-Base",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-14B-Base",
    "unsloth/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-11B-Vision",
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    "unsloth/Meta-Llama-3.1-8B",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/DeepSeek-R1-0528-Qwen3-8B",
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "unsloth/DeepSeek-R1-Distill-Qwen-14B",
    "unsloth/DeepSeek-R1-Distill-Llama-8B",
    "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    "unsloth/DeepSeek-R1-Distill-Qwen-14B",
    "unsloth/DeepSeek-R1-Distill-Llama-8B",
    "unsloth/Mistral-Nemo-Base-2407",
    "unsloth/Mistral-Nemo-Instruct-2407",
    "unsloth/gemma-2-2b",
    "unsloth/gemma-2-2b-it",
    "unsloth/gemma-2-9b",
    "unsloth/gemma-2-9b-it",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "ibm-granite/granite-3.1-2b-base",
    "ibm-granite/granite-3.1-2b-instruct",
    "ibm-granite/granite-3.1-8b-base",
    "ibm-granite/granite-3.1-8b-instruct",
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
]

# NOTE TO USERS: Update this path to point to your corpus of text files.
allfiles = glob.glob('*.txt')

results = {}

# Main evaluation loop over all defined LLMs
for model_id in model_ids:
    print('Starting ', model_id)
    gc.collect()
    torch.cuda.empty_cache()

    # Configure 4-bit quantization to fit multiple models in limited VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Change to torch.float16 if bfloat16 is not available
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16 # Also set overall torch_dtype for non-quantized parts
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    results[model_id] = defaultdict(list)

    # Process every text file in the dataset using the current model
    for textfile in tqdm(allfiles):
        path, filename = os.path.split(textfile)
        
        # Parses the ground-truth generation temperature from the filename.
        # Assumes filenames are structured such that characters [-8:-5] 
        # (e.g., '0.7' or '1.0' prior to '.txt') represent the float temperature.
        temp = float(filename[-8:-5])
        
        # Extracts the name of the model that generated the text from the filename
        gmodel = filename[:-10]
        
        with open(textfile, encoding='utf-8') as file:
            pretext = file.read()
            if len(pretext)>maxlen:
                text = pretext[:maxlen]
            else:
                text = pretext
            
            # To estimate the temperature of a text, we first compute the logits
            # for all the text and then solve for maximum likelyhood temperature
            logits, tokens = compute_logits_from_text(model, tokenizer, text)
            estimatedr_T = estimate_temperature_fast(logits, tokens)

            results[model_id][gmodel].append((temp, estimatedr_T))
            
    # Save the estimation results for this model to a JSON file
    # Uses the repository name part of the model_id (e.g., 'Qwen3-8B.json')
    filen = model_id[model_id.find('/')+1:]+'.json'
    with open(filen, 'w') as json_file:
        json.dump(results, json_file, indent=4)
