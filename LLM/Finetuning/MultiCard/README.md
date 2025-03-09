# Optimum-Habana DPO Training Guide

## Installation

To use the example associated with the latest stable release, run:
```bash
git clone https://github.com/huggingface/optimum-habana
pip install -e ./optimum-habana
```

To install the requirements for every example:
```bash
pip install -r examples/trl/requirements.txt
```


## Supervised Finetuning

1. Supervised fine-tuning of the mistralai/Mixtral-8x7B-Instruct-v0.1 on **4 cards**:
```bash
DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 4 --use_deepspeed sft.py \
    --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --dataset_name "philschmid/dolly-15k-oai-style" \
    --subset 'data/' \
    --streaming False \
    --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
    --output_dir="./model_mixtral" \
    --do_train \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=100 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --lora_target_modules "q_proj" "v_proj" \
    --bf16 \
    --remove_unused_columns=False \
    --max_seq_length 512 \
    --run_name="sft_mixtral" \
    --report_to=none \
    --use_habana \
    --use_lazy_mode
```

## DPO Pipeline

### Training

#### For meta-llama/Llama-2-70b-hf

The following example demonstrates the creation of StackLlaMa 2: a Stack Exchange Llama-v2-70b model. The DPO training process involves two main steps.

For large models like Llama2-70B, we can use DeepSpeed Zero-3 to enable DPO training across multiple cards:

#### Step 1: Supervised Fine-Tuning (SFT)

First, we perform supervised fine-tuning of the base Llama-v2-70b model to create Llama-v2-70b-se:

**Option A: Using DeepSpeed for distributed training:**

```bash
DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed sft.py \
        --model_name_or_path meta-llama/Llama-2-70b-hf \
        --dataset_name "lvwerra/stack-exchange-paired" \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --output_dir="./sft" \
        --do_train \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=100 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_steps=100 \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --lora_target_modules "q_proj" "v_proj" \
        --bf16 \
        --remove_unused_columns=False \
        --run_name="sft_llama2" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
```

**Option B: Using MPI for distributed training:**

```bash
DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 8 --use_mpi sft.py \
        --model_name_or_path meta-llama/Llama-2-70b-hf \
        --dataset_name "lvwerra/stack-exchange-paired" \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --output_dir="./sft" \
        --do_train \
        --max_steps=500 \
        --logging_steps=10 \
        --save_steps=100 \
        --per_device_train_batch_size=1 \
        --per_device_eval_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_steps=100 \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --lora_target_modules "q_proj" "v_proj" \
        --bf16 \
        --remove_unused_columns=False \
        --run_name="sft_llama2" \
        --report_to=none \
        --use_habana \
        --use_lazy_mode
```

To merge the adaptors and get the final SFT merged checkpoint, use the `merge_peft_adapter.py` helper script that comes with TRL:

```bash
python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-70b-hf" --adapter_model_name="sft" --output_name="sft/final_merged_checkpoint"
```

### DeepSpeed vs MPI for Distributed Training

**DeepSpeed:**  
A deep learning optimization library designed for efficient distributed training of large models. Key features include:

- Zero Redundancy Optimizer (ZeRO) with three stages of memory optimization
- Partitioning of optimizer states, gradients, and model parameters across GPUs
- Mixed-precision training, gradient accumulation, and efficient communication primitives
- Seamless integration with PyTorch requiring minimal code changes

Resources:
- [DeepSpeed GitHub Repository](https://github.com/deepspeedai/DeepSpeed)
- [ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)

**MPI (Message Passing Interface):**  
A general-purpose communication protocol and library for parallel computing that:

- Provides infrastructure for GPUs or nodes to communicate with each other
- Handles sending and receiving of messages (gradients, model parameters) between processes
- Requires more explicit management of data partitioning and model distribution
- Offers more control (and responsibility) over distributed training details

#### Step 2: DPO Training

Run the DPO trainer using the model saved from the previous step:

```bash
DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 python ../gaudi_spawn.py --world_size 8 --use_deepspeed dpo.py \
        --model_name_or_path="sft/final_merged_checkpoint" \
        --tokenizer_name_or_path=meta-llama/Llama-2-70b-hf \
        --deepspeed ../language-modeling/llama2_ds_zero3_config.json \
        --lora_target_modules "q_proj" "v_proj" "k_proj" "out_proj" "fc_in" "fc_out" "wte" \
        --output_dir="dpo" \
        --max_prompt_length=256 \
        --max_length=512 \
        --report_to=none
```

### Merging the Adaptors

To merge the adaptors into the base model, use the `merge_peft_adapter.py` helper script from TRL:

```bash
python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-70b-hf" --adapter_model_name="dpo" --output_name="stack-llama-2"
```

This will also push the model to your HuggingFace Hub account.

### Running the Model

Load the DPO-trained LoRA adaptors saved by the DPO training step and run it through the [text-generation example](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation):

```bash
python ../gaudi_spawn.py --world_size 8 --use_deepspeed run_generation.py \
--model_name_or_path ../trl/stack-llama-2/ \
--use_hpu_graphs --use_kv_cache --batch_size 1 --bf16 --max_new_tokens 100 \
--prompt "Here is my prompt"
```

## Fine-Tuning Command and Parameters

### Environment Variable
1. `DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1`: Environment variable for DeepSpeed configuration

### Command Parameters
1. `python ../gaudi_spawn.py --world_size 8 --use_deepspeed`: Launches training with DeepSpeed on 8 HPUs
2. `sft.py`: The main training script

### Script Arguments
| Flag | Description | Value |
|------|-------------|-------|
| `--model_name_or_path` | Base model to be fine-tuned | `meta-llama/Llama-2-70b-hf` |
| `--dataset_name` | Training dataset to use | `"lvwerra/stack-exchange-paired"` |
| `--deepspeed` | DeepSpeed configuration file path | `../language-modeling/llama2_ds_zero3_config.json` |
| `--output_dir` | Directory to save the fine-tuned model | `"./sft"` |
| `--do_train` | Enables training mode | Enabled |
| `--max_steps` | Maximum number of training steps | `500` |
| `--logging_steps` | Frequency of logging during training | `10` |
| `--save_steps` | Frequency of saving model checkpoints | `100` |
| `--per_device_train_batch_size` | Batch size per HPU for training | `1` |
| `--per_device_eval_batch_size` | Batch size per HPU for evaluation | `1` |
| `--gradient_accumulation_steps` | Steps to accumulate gradients before updating | `2` |
| `--learning_rate` | Initial learning rate for training | `1e-4` |
| `--lr_scheduler_type` | Learning rate scheduler type | `"cosine"` |
| `--warmup_steps` | Warmup steps for the learning rate scheduler | `100` |
| `--weight_decay` | Weight decay for regularization | `0.05` |
| `--optim` | Optimizer type | `"paged_adamw_32bit"` |
| `--lora_target_modules` | Modules for LoRA application | `"q_proj", "v_proj", "k_proj", "o_proj"` |
| `--bf16` | Enables bfloat16 mixed precision training | Enabled |
| `--remove_unused_columns` | Keeps all dataset columns | `False` |
| `--run_name` | Training run name | `"sft_llama2"` |
| `--report_to` | Disables external reporting | `none` |
| `--use_habana` | Enables Habana Gaudi accelerator support | Enabled |
| `--use_lazy_mode` | Enables lazy mode for memory optimization | Enabled |

Adjust these parameters based on your specific hardware capabilities, dataset size, and training objectives. The DeepSpeed Zero-3 configuration enables efficient training of large models like Llama-2-70b across multiple HPUs.