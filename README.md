# Estimating Text Temperature with Language Models

This repository contains the companion code for the paper *"Estimating Text Temperature with Language Models"*

# Overview

Autoregressive language models typically use a temperature parameter ($T$) at inference to shape the probability distribution and control the randomness (or "creativity") of the generated text. Once a text has been generated, or when dealing with a human-written text, this parameter is unknown.

This repository provides a methodology and script to estimate **the temperature of any text** — including human-written corpora — with respect to a given language model. This is achieved by calculating the Maximum Likelihood Estimate (MLE) of the temperature parameter, finding the temperature at which the sum of the observed logits equals the sum of the expected logits predicted by the model.

## Potential applications of this technique include:

- Domain-Adaptive Decoding: Dynamically adjusting generation temperature based on the estimated "intrinsic temperature" of the prompt's domain.
- Model Forensics and Detection: Identifying hallucinated or poorly generated code/text based on anomalous estimated temperatures.
- Quantifying Over-Alignment: Using temperature estimation failure as a diagnostic for distributional collapse caused by aggressive RLHF or distillation.

# Requirements

The code is written in Python and heavily utilizes PyTorch and Hugging Face transformers. The use of bitsandbytes is included to load models in 4-bit precision, allowing evaluation of larger models on consumer hardware.
```
pip install torch numpy scipy transformers bitsandbytes tqdm
```

# Usage

The main script is esttemp_.py. It is designed to iterate over a local directory of text files, run them through a wide selection of small-to-medium Large Language Models (LLMs), and output the estimated temperatures.

## 1. Prepare your data

By default, the script looks for .txt files in d:/temperatures/. You must update the glob.glob() path in esttemp_.py to point to your corpus.
```
# Update this line in esttemp_.py:
allfiles = glob.glob('*.txt')
```

Note on Filenames: If you are evaluating generated texts and wish to compare the estimated temperature against the actual generation temperature, the script expects the filename to contain the generation temperature near the end (specifically, characters [-8:-5], e.g., modelname_1.5.txt). Modify the parsing logic in the script if your naming convention differs.

## 2. Run the estimation

Execute the script to begin processing.
```
python esttemp_.py
```

The script will automatically load each model in 4-bit precision (to manage VRAM), process the texts, and save the results for each model as a JSON file (e.g., Qwen3-14B.json) in the working directory.

# Citation

If you use this code or methodology in your research, please cite the corresponding paper:

```
@misc{mikhaylovskiy2026estimating,
  title={Estimating Text Temperature with Language Models},
  author={Mikhaylovskiy, Nikolay},
  journal={arXiv preprint},
  eprint={2601.02320},
  archivePrefix={arXiv},
  year={2026}
}
```
