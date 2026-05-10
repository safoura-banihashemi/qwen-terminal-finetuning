# Qwen3-0.6B · Terminal Command Generation via GRPO

A reinforcement learning fine-tuned model that generates Linux terminal commands
from natural language descriptions. Trained using **GRPO (Group Relative Policy
Optimization)** on a custom terminal task environment inspired by CAMEL-AI's SETA framework.

---

## Results

| | Base Qwen3-0.6B | Fine-tuned (this model) | Gain |
|---|---|---|---|
| Accuracy | 13.6% (3/22) | **72.7% (16/22)** | **+59.1%** |
| Mean Reward | 0.136 | 0.841 | +0.705 |
| Correct tasks | 3 | 16 | +13 tasks |
| Partial credit | 0 | 5 | — |
| Wrong | 19 | 1 | -18 |

> The base model was evaluated by disabling LoRA adapter layers on the same
> model instance, ensuring a fair apples-to-apples comparison.

---

## What is GRPO?

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm
that trains a model without any labeled data. Instead of learning from
correct/incorrect labels, the model:

1. Generates multiple candidate commands for each task (`NUM_GENERATIONS=4`)
2. Each candidate is scored by a reward function
3. Candidates that scored higher than the group average get reinforced
4. The model gradually learns which command patterns get rewarded

No human labels needed — the model learns purely from trying and receiving feedback.

---

## Inspiration: CAMEL-AI's SETA Framework

This project is directly inspired by **SETA (Scaling Environments for Terminal
Agents)**, released by CAMEL-AI. SETA trains RL agents on terminal tasks using
Docker containers with real shell execution and verified test suites.

This project reimplements the **core SETA concept** in a lightweight,
single-notebook format:

| | SETA (original) | This project |
|---|---|---|
| Base model | Qwen3-8B | Qwen3-0.6B |
| Training method | RLVR / GRPO | GRPO |
| Environment | Docker + real shell | Lightweight Python gym |
| Task verification | pytest + run-tests.sh | Regex/string matching |
| Number of tasks | 400 | 22 |
| Hardware | AWS fleet | Single A100 GPU |

The key difference: SETA executes commands in real Docker containers and
verifies actual output. This project uses regex-based reward scoring, which
is simpler but sufficient to demonstrate that GRPO learns terminal command
generation effectively even at the 0.6B scale.

---

## Task Categories

The model was trained on 22 terminal tasks across 6 categories:

| Category | Tasks |
|---|---|
| File system | ls, mkdir, find, cp, rm, chmod |
| Text processing | grep, wc, sort, sed, tail |
| Process / System | ps, kill, df, free |
| Network | curl, ping |
| Git | clone, log, diff |
| Python / pip | pip install, python run |

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B',
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base, 'safoura00/qwen3-terminal-grpo')
tokenizer = AutoTokenizer.from_pretrained('safoura00/qwen3-terminal-grpo',
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

def ask_terminal(task: str) -> str:
    system = (
        'You are a Linux terminal expert. Given a task description, respond with '
        'ONLY the exact terminal command -- no explanation, no markdown, no extra text.'
    )
    prompt = f'{system}\n\nTask: {task}\nCommand:'
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
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

# Examples
print(ask_terminal('List all files including hidden ones'))
# → ls -la

print(ask_terminal('Show the last 5 git commits in one line each'))
# → git log --oneline -5

print(ask_terminal('Find all Python files recursively'))
# → find . -name "*.py"

print(ask_terminal('Forcefully kill process with PID 1234'))
# → kill -9 1234
```

---

## Training Configuration

```python
MODEL_ID        = 'Qwen/Qwen3-0.6B'
NUM_EPISODES    = 500
BATCH_SIZE      = 16
NUM_GENERATIONS = 4       # rollouts per prompt
MAX_NEW_TOKENS  = 64
LR              = 5e-6
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
WARMUP_STEPS    = 20
MAX_GRAD_NORM   = 0.3
```

**Reward function:**
- `1.0` — exact match or matches accepted regex pattern
- `0.5` — correct base command but wrong flags
- `0.0` — wrong command entirely

---

## Benchmark Results

![Benchmark Chart](benchmark_chart.png)
![Training Curves](training_curves.png)

**Best performing tasks (fully correct):**
`ls-l`, `mkdir`, `find-py`, `cp-r`, `rm-rf`, `chmod`, `awk-sum`,
`tail-f`, `kill-9`, `curl-url`, `ping`, `git-clone`, `git-log`,
`git-diff`, `pip-install`, `python-run`

**Partial credit (correct command, wrong flags):**
`grep-r`, `wc-l`, `sort-u`, `ps-aux`, `df-h`, `free-m`

---

## Limitations

- Trained on 22 specific task types — may not generalize to unseen commands
- Reward function uses regex matching, not real shell execution
- 0.6B parameter model — larger models would achieve higher accuracy
- Single epoch of training — more episodes would likely push accuracy higher

---

## Training Infrastructure

- **Hardware:** NVIDIA A100 40GB (Google Colab Pro)
- **Training time:** ~6 hours
- **Framework:** HuggingFace TRL (GRPOTrainer)
- **Quantization:** 4-bit NF4 (bitsandbytes)
- **Adapter:** LoRA via PEFT

---

## Links

- Training notebook: https://github.com/safoura-banihashemi/qwen3-terminal-grpo
- Dataset: safoura00/qwen3-terminal-tasks
- Base model: https://huggingface.co/Qwen/Qwen3-0.6B

### Links
- SETA GitHub: https://github.com/camel-ai/seta
- SETA Dataset: https://huggingface.co/datasets/camel-ai/seta-env
