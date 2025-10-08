# WinFLoRA: Incentivizing Client-Adaptive Aggregation in Federated LoRA under Privacy Heterogeneity

The code of  WinFLoRA is presented here comprehensively and systematically.

### 1. Environments and LLMs:

#### Large Language Models:

We adopt Llama3.2 (1B), TinyLlama, Gemma3 (1B), GPT2-Large, and OPT (1.3B) for evaluation in our experiments.

#### Language Version:

Python 3.12.9

#### Required Packages:

Main libraries include torch 2.6.0, transformers 4.50.1, and tokenizers 0.21.1, etc. The full package list can be found in requirements.txt/.yaml file.

### 2. Quick Set-up:

Install environments with requirements.txt:

```
conda/pip install -r requirements.txt
```

or using requirements.yaml:

```
conda env create -f requirements.yaml
```

### 3. Additional Experimental Setup:

We supplement a detailed experimental setup, including dataset information and LLMs in our evaluation ([View the Setup](./Experiment_setup.md)).

