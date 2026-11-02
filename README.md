# Fine-Tuning Llama 3.1 8B for Bengali Empathetic Conversations

A parameter-efficient fine-tuning project using LoRA and Unsloth to create an empathetic conversational AI in Bengali language.

## 📋 Table of Contents
- [Overview](#overview)
- [Configuration Choices](#configuration-choices)
- [Training Strategy](#training-strategy)
- [Challenges Faced](#challenges-faced)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Overview

This project fine-tunes Meta's Llama 3.1 8B Instruct model on a Bengali empathetic conversations corpus. The goal is to create a model that can respond empathetically to user queries in Bengali, maintaining cultural and linguistic nuances.

**Dataset**: Bengali empathetic conversation corpus (Questions & Answers). Source, version, and license are not recorded in this repository; see [NOTICE.md](./NOTICE.md).  
**Base Model**: unsloth/Meta-Llama-3.1-8B-Instruct  
**Training Technique**: LoRA (Low-Rank Adaptation) with Unsloth optimization  
**Hardware**: 2x Tesla T4 GPUs (15.8GB each)  
**Status**: Training completed successfully; inference limited by memory constraints

## Configuration Choices

### Why LoRA?

**LoRA (Low-Rank Adaptation)** was chosen for several critical reasons:

1. **Parameter Efficiency**: Instead of fine-tuning all 8 billion parameters, LoRA trains only ~1% through low-rank adapter matrices
2. **Memory Savings**: Drastically reduces VRAM requirements, making training feasible on consumer GPUs
3. **Prevents Catastrophic Forgetting**: Base model knowledge is preserved while adapting to the specific task
4. **Faster Training**: Fewer parameters to update means faster iterations
5. **Easy Deployment**: Adapters can be loaded/unloaded, allowing multiple specializations of the same base model

### LoRA Configuration Parameters

```python
r = 16                    # LoRA rank (attention dimension)
lora_alpha = 16          # Scaling factor for LoRA updates
lora_dropout = 0         # No dropout for optimal performance
bias = "none"            # Don't train bias parameters
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj"       # MLP layers
]
```

**Rationale for r=16**: 
- Higher ranks (32, 64) provide better quality but increase trainable parameters exponentially
- Lower ranks (4, 8) are too constrained for complex language tasks
- r=16 strikes the optimal balance between quality and efficiency for instruction tuning

**Why target all attention and MLP layers?**
- Attention layers control how the model focuses on different parts of the input
- MLP layers handle the transformation and reasoning
- Targeting both ensures comprehensive adaptation to Bengali empathetic responses

### Why Unsloth?

**Unsloth** is a highly optimized library that provides significant advantages:

1. **2x Faster Training**: Optimized CUDA kernels for transformer operations
2. **30% Less VRAM**: Custom gradient checkpointing implementation
3. **2x Larger Batch Sizes**: Memory efficiency enables larger effective batch sizes
4. **Drop-in Replacement**: Compatible with HuggingFace transformers
5. **2-GPU Model Sharding**: Loads the model across both T4s with `device_map="balanced"` (model parallelism, not data-parallel DDP)

### Model Loading Configuration

```python
max_seq_length = 2020      # Maximum context length
dtype = None               # Auto-detect (FP16 for T4, BF16 for Ampere+)
load_in_4bit = False       # Full precision for quality (can enable for memory)
device_map = "balanced"    # Shard the model across the two T4 GPUs (value used in the run)
```

**Why max_seq_length=2020?**
- Bengali text can be verbose; empathetic responses need context
- 2020 tokens accommodate question + system prompt + detailed answer
- Below 4096 limit, avoiding excessive memory usage

**Why not 4-bit quantization?**
- Full precision used during training for optimal quality
- 4-bit quantization recommended for inference to reduce memory usage
- Can be enabled for single GPU deployment

**Device Map**: The logged run used `device_map="balanced"`, which shards model layers across both GPUs. Unsloth reports `Data Parallel GPUs = 1`, so this is model parallelism rather than DDP; the effective batch of 16 is the total, not per GPU.

## Training Strategy

### Data Preparation

**Split Strategy:**
- **Training**: 80% (30,568 samples)
- **Validation**: 10% (3,821 samples)
- **Test**: 10% (3,821 samples)

Two-stage split ensures proper evaluation without data leakage. Validation monitors training progress, test evaluates final performance.

**Data Cleaning:**
- Remove missing values and empty strings
- Strip whitespace for consistency
- Validate Bengali text encoding
- Filter out corrupted samples

### Prompt Format

Following Llama 3.1's official chat template:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a sympathetic and helpful assistant. You answer people's questions in Bengali language.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_response}
<|eot_id|>
```

This structured format helps the model distinguish roles and maintain conversation flow.

### Hyperparameter Selection

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
effective_batch_size = 16          # 2 * 8 = 16 total across the two sharded GPUs

learning_rate = 2e-4               # Standard for instruction fine-tuning
max_steps = 1000                   # Fixed optimizer steps; matches the logged run length
warmup_steps = 100
weight_decay = 0.01
lr_scheduler_type = "cosine"

optimizer = "adamw_8bit"           # Memory-efficient optimizer
```

**Batch Size Strategy:**
- Small per-device batch size (2) prevents OOM on 15.8GB GPUs
- Gradient accumulation (8) simulates larger batches for stable training
- Effective batch size of 16 total (2 per device × 8 accumulation) provides stable gradient estimates
- The model is sharded across the two T4s (model parallelism), not data-parallel replicated
- Follows best practices: batch_size=2 (VRAM driver), accumulation=8 (time driver)

**Learning Rate (2e-4):**
- Standard for instruction fine-tuning with LoRA
- Balanced between fast convergence and stability
- Works well with 8-bit optimizer
- Higher than full fine-tuning (5e-5) due to fewer trainable parameters

**1000 Steps (~0.52 epoch):**
- The code fixes `max_steps=1000`, the number of optimizer steps the logged run actually reached
- With an effective batch of 16, one pass over the 30,568-sample training split is ~1,911 steps, so 1,000 steps is ~0.52 epoch
- `load_best_model_at_end` keeps the best checkpoint by eval loss; the best logged eval loss is 0.5519 at step 950
- Training was not stopped early: no early-stopping callback is configured, and the last logged step is 1,000

**Warmup Steps (100):**
- Gradual learning rate increase prevents early training instability
- Especially important when starting from instruction-tuned model
- Represents ~10% of the 1,000 total training steps

**Cosine Learning Rate Schedule:**
- Smooth decay from peak to near-zero
- Better than linear decay for fine-tuning
- Allows model to settle into local minima

### Evaluation Strategy

```python
eval_strategy = "steps"
eval_steps = 50                    # Evaluate every 50 training steps
save_steps = 100                   # Checkpoint every 100 steps
save_total_limit = 3               # Keep only best 3 checkpoints
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
```

**Frequent Evaluation (every 50 steps):**
- Catch overfitting early
- Monitor loss curves in real-time via W&B
- Fine-grained performance tracking

**Checkpoint Management:**
- Save every 100 steps for recovery
- Keep only 3 best to conserve disk space
- Load best model at end ensures optimal final model

**Note**: Updated `gradient_accumulation_steps` from 4 to 8 following best practices, providing an effective batch size of 16 (2 batch size × 8 accumulation).

### Gradient Checkpointing

```python
use_gradient_checkpointing = "unsloth"
```

Unsloth's custom implementation trades computation for memory:
- Recomputes activations during backward pass instead of storing
- 30% memory reduction vs standard checkpointing
- Essential for long sequences (2020 tokens)

## Challenges Faced


### 1. Gradient Accumulation Configuration

**Problem**: Initial configuration had suboptimal effective batch size.

**Root Cause**: 
- Started with `gradient_accumulation_steps=4`
- Effective batch size was only 8 (2 × 4)
- Below recommended range of 16-32 for stable training

**Solution**:
- Updated `gradient_accumulation_steps` from 4 to 8
- New effective batch size: 16 (2 × 8)
- Follows best practices: batch_size=2 (VRAM driver), accumulation=8 (time driver)

**Impact**: More stable gradient estimates and smoother convergence.

### 2. Loss Plateau at ~0.6

**Problem**: Training loss fell from 1.67 to ~0.63 within the first ~80 steps, then stayed in the 0.58-0.63 range for the rest of the run.

**Observation**:
- Most of the loss reduction happens before step 100 (train loss 0.629 at step 100)
- Extending training to 1,000 steps lowered train loss to 0.579 (step 1000); the best eval loss was 0.5519 at step 950
- After ~step 100 the curve is flat and further steps give only marginal improvement

**Decision**:
- Ran a fixed `max_steps=1000` instead of a short 100-step run
- Kept the learning rate at 2e-4 (standard for LoRA fine-tuning)

**Impact**: The longer run produced a stable, well-checkpointed model, but the plateau was not sharply "broken" at a later step; most learning had already occurred by ~step 100.

**Lesson**: In this run the curve converges early. Report step-wise gains honestly instead of attributing a late breakthrough.

### 3. Memory Management with Large Model

**Problem**: 8B parameter model with long sequences (2020 tokens) pushes memory limits on 2x T4 GPUs.

**Solutions Implemented**:
- Model sharding across 2 GPUs with `device_map="balanced"` (model parallelism, not DDP)
- Gradient checkpointing with Unsloth optimization
- Small per-device batch size (2) with gradient accumulation (8)
- 8-bit optimizer (AdamW-8bit) reduces optimizer state memory
- Learning rate 2e-4, matching the logged training config

**Inference Limitation**: 
- Memory constraints prevent full inference on test set during training session
- Model saved successfully for offline inference
- Recommend 4-bit quantization (QLoRA) for production inference

**Lesson**: Training and inference have different memory requirements; consider quantization for deployment.

## Results

### Training Status

✅ **Training Completed**
- Trained for 1,000 optimizer steps (`max_steps=1000`; ~0.52 epoch of the 30,568-sample training split)
- LoRA adapter and merged 16-bit model were written to Kaggle working directories, not committed to this repository
- Training logs available in Weights & Biases (run: `llama-3.1-8b-finetuning-v1`)

### Training Metrics

**Loss Progression** (from the W&B export in `reports/CSV`):
- **Initial Training Loss** (Step 1): 1.67
- **Initial Eval Loss** (Step 10): 1.56
- **Converged by ~Step 100**: training loss 0.629, eval loss 0.625
- **Later gains were marginal**: training loss 0.586 at step 300, 0.567 at step 950
- **Last Logged Step** (Step 1000): training loss 0.579, eval loss 0.565
- **Best Eval Loss**: 0.5519 at step 950

**Gradient Norm Stability:**
- Initial gradient norm: 1.39
- Stable range throughout training: 0.22-0.36
- Indicates healthy gradient flow and no exploding/vanishing gradients

**Learning Rate Schedule:**
- Warmup phase (Steps 0-100): Linear increase from 0 to 2e-4
- Training phase (Steps 100-1000): Cosine decay from 2e-4 to ~6e-10 by step 1000
- Successfully prevented early instability and allowed smooth convergence

**Training Configuration**: 
- **Total Steps**: 1,000 (`max_steps=1000`)
- **Batch size**: 2 per device
- **Gradient accumulation**: 8 steps
- **Effective batch size**: 16 total
- **Peak learning rate**: 2e-4
- **Warmup steps**: 100
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Cosine
- **Hardware**: 2x Tesla T4 GPUs, model sharded with `device_map="balanced"`

**Key Observations:**
- Total training loss fell from 1.67 to 0.579 at step 1000 (~65%); most of the drop (1.67 → 0.635) happened in the first 80 steps
- Eval loss closely tracked training loss (no overfitting); best eval loss 0.5519 at step 950
- Gains after ~step 100 were marginal: training loss went from 0.629 at step 100 to 0.579 at step 1000
- Stable gradient norms indicate proper hyperparameter tuning

### Evaluation Status

⚠️ **Not run**

The evaluation cells (BLEU, ROUGE, perplexity) are commented out in the notebook and were not executed for this run. Every quality statement above is based on training/eval loss only.
- Full evaluation on the 3,821-row test split was not completed
- To run it: install `rouge_score` (now in `requirements.txt` and the install cell), uncomment the evaluation cells, and record the output
- Model successfully saved and can be evaluated offline

**For full evaluation**, load the saved model with 4-bit quantization:
```python
fine_tuner = LLAMAFineTuner(
    model_name="llama-3.1-8b-bangla-empathic-lora",
    load_in_4bit=True  # Enable for inference
)
```

### Model Artifacts

The training run wrote these to Kaggle-ephemeral working directories. They are not included in this repository, and no Hugging Face Hub copy is linked, so the saved weights cannot be verified from the repo alone.
- **LoRA Adapters**: `llama-3.1-8b-bangla-empathic-lora/` (Kaggle working directory)
- **Merged Model**: `llama-3.1-8b-bangla-empathic-merged/` (Kaggle working directory)
- **Training Logs**: Weights & Biases project `llama-bangla-empathic`
- **Checkpoints**: kept only inside the training session (`outputs/`)

## Usage

### Loading the Fine-tuned Model

**Option 1: Full Precision (Requires ~16GB VRAM)**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama-3.1-8b-bangla-empathic-lora",
    max_seq_length=2020,
    dtype=None,
    load_in_4bit=False,
)
```

**Option 2: 4-bit Quantization (Recommended - Requires ~5GB VRAM)**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama-3.1-8b-bangla-empathic-lora",
    max_seq_length=2020,
    dtype=None,
    load_in_4bit=True,  # Enable 4-bit for memory efficiency
)
```

### Inference

```python
# Enable inference mode for faster generation
FastLanguageModel.for_inference(model)

# Format prompt using Llama 3.1 chat template
question = "আমি খুব চিন্তিত এবং মানসিক চাপে আছি।"
prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a sympathetic and helpful assistant. You answer people's questions in Bengali language.\n<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    f"{question}\n<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    use_cache=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Extract assistant's response
response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
print(response)
```

### Batch Inference

```python
questions = [
    "আমি খুব একা অনুভব করছি।",
    "আমার জীবনে কোন উদ্দেশ্য নেই।",
    "আমি সবসময় ব্যর্থ হই।"
]

for question in questions:
    # Format and generate response
    # ... (same as above)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

## Dependencies

Install the pinned environment recorded for the training run:

```bash
pip install -r requirements.txt
```

## License and Notices

The base model and any LoRA adapter or merged weights produced by this project are governed by the Llama 3.1 Community License (Copyright © Meta Platforms, Inc.). Dataset terms are set by the corpus publisher. See [NOTICE.md](./NOTICE.md) for the required attribution and notices.
