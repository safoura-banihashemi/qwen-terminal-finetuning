# Qwen3 RL Terminal Agent

A **'Qwen/Qwen3-0.6B'** model fine-tuned with **GRPO (Group Relative Policy Optimization)**  
to generate correct Linux terminal commands from natural-language descriptions.

## Training

| Parameter | Value |
|-----------|-------|
| Base model | `'Qwen/Qwen3-0.6B'` |
| Method | GRPO (RL) + LoRA |
| Episodes | {NUM_EPISODES} |
| LoRA rank | {LORA_R} |
| Gym tasks | {len(gym.tasks)} |

## Benchmark

| Metric | Score |
|--------|-------|
| Accuracy | {br['accuracy']:.1%} |
| Mean reward | {br['mean_reward']:.3f} |
| Fully correct | {br['fully_correct']}/{br['total_tasks']} |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained('{MODEL_ID}')
model = PeftModel.from_pretrained(base, '{HF_REPO}')
tokenizer = AutoTokenizer.from_pretrained('{HF_REPO}')

prompt = """You are a Linux terminal expert. Given a task, output ONLY the command.

Task: List all files including hidden ones
Command:"""

inputs = tokenizer(prompt, return_tensors='pt')
out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
# → ls -la
