# Qwen3-0.6B Terminal Command Fine-tuning via GRPO

A reinforcement learning fine-tuned model that generates Linux terminal commands
from natural language descriptions. Trained using GRPO (Group Relative Policy 
Optimization) on a custom terminal task environment.

## Model Details

| | |
|---|---|
| Base model | Qwen/Qwen3-0.6B |
| Fine-tuning method | GRPO (Reinforcement Learning) |
| Adapter type | LoRA (r=16, alpha=32) |
| Training hardware | A100 GPU (40 GB) |
| Episodes | 500 |
| Tasks | 22 Linux terminal task categories |
| Accuracy | 31.8% (7/22 fully correct, 4 partial) |

## What is GRPO?

GRPO (Group Relative Policy Optimization) is the same RL algorithm used to train 
DeepSeek-R1. Instead of learning from labeled examples, the model learns by:
1. Generating multiple command candidates for each task
2. Scoring each candidate with a reward function
3. Reinforcing commands that got higher rewards

No human labels needed — the model learns purely from trying and getting feedback.

## Task Categories

The model was trained on 22 terminal task types across 6 categories:

- **File system** — ls, mkdir, find, cp, rm, chmod
- **Text processing** — grep, wc, sort, sed, tail
- **Process/System** — ps, kill, df, free
- **Network** — curl, ping
- **Git** — clone, log, diff
- **Python/pip** — pip install, python run

## Benchmark Results

| Metric | Score |
|---|---|
| Accuracy | 31.8% |
| Mean reward | 0.409 |
| Fully correct | 7 / 22 |
| Partial credit | 4 / 22 |
| Wrong | 11 / 22 |

**Best performing tasks:** git-clone, git-log, git-diff, rm-rf, chmod, grep-r, kill-9

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
model = PeftModel.from_pretrained(base, 'safoura00/qwen-terminal-finetuning')
tokenizer = AutoTokenizer.from_pretrained('safoura00/qwen-terminal-finetuning')

def ask_terminal(task: str) -> str:
    prompt = (
        "You are a Linux terminal expert. Given a task description, respond with "
        "ONLY the exact terminal command -- no explanation, no markdown, no extra text.\n\n"
        f"Task: {task}\nCommand:"
    )
    inputs = tokenizer(prompt, return_tensors='pt')
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

print(ask_terminal('List all files including hidden ones'))
# → ls -la

print(ask_terminal('Show the last 5 git commits'))
# → git log --oneline -5

print(ask_terminal('Find all Python files recursively'))
# → find . -name "*.py"
```

## Training Setup

```python
# Key configuration
MODEL_ID        = 'Qwen/Qwen3-0.6B'
NUM_EPISODES    = 500
BATCH_SIZE      = 16
NUM_GENERATIONS = 4       # rollouts per prompt
LR              = 1e-5
LORA_R          = 16
LORA_ALPHA      = 32
MAX_NEW_TOKENS  = 64
```

The reward function scores each generated command:
- **1.0** — exact match or matches accepted pattern
- **0.5** — correct base command but wrong flags
- **0.0** — wrong command entirely

## Inspired by the SETA Framework

This project is directly inspired by **SETA (Scaling Environments for Terminal Agents)**,
an open-source framework released by CAMEL-AI.

### What SETA is

SETA is a toolkit and environment stack built specifically for training AI agents that
operate inside a Unix-style shell.

This project reimplements the **core idea of SETA** — using GRPO reinforcement learning
to teach a small language model terminal commands.

### Links
- SETA GitHub: https://github.com/camel-ai/seta
- SETA Dataset: https://huggingface.co/datasets/camel-ai/seta-env
