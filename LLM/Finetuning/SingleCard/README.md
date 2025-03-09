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

1. The following example is for the supervised Lora finetune with Qwen2 model for conversational format dataset.
```bash
python sft.py \
    --model_name_or_path "Qwen/Qwen2-7B" \
    --dataset_name "philschmid/dolly-15k-oai-style" \
    --streaming False \
    --bf16 True \
    --subset '' \
    --output_dir ./model_qwen \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 3e-4 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "cosine" \
    --max_grad_norm  0.3 \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --use_peft True \
    --lora_r 4 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --max_seq_length 512 \
    --adam_epsilon 1e-08 \
    --use_flash_attention
```

### Command Parameters
` python sft.py `: Launches the main training script.


### Script Arguments
| Flag                         | Description                                                                 | Value                                       |
|----------------------------------|---------------------------------------------------------------------------|---------------------------------------------|
| `--model_name_or_path`           | Pretrained model identifier or path to local model                        | `"Qwen/Qwen2-7B"`                          |
| `--dataset_name`                 | Name of the dataset to use for training                                   | `"philschmid/dolly-15k-oai-style"`         |
| `--streaming`                    | Whether to stream the dataset instead of loading it entirely              | `False`                                    |
| `--bf16`                         | Use bfloat16 precision for training                                      | `True`                                     |
| `--subset`                       | Subset of the dataset to use                                             | `''` (empty string)                        |
| `--output_dir`                   | Directory to save the trained model                                      | `./model_qwen`                             |
| `--num_train_epochs`             | Number of training epochs                                                | `1`                                        |
| `--per_device_train_batch_size`  | Batch size per device for training                                      | `16`                                       |
| `--eval_strategy`                | Evaluation strategy (e.g., `"no"`, `"epoch"`, `"steps"`)                 | `"no"`                                     |
| `--save_strategy`                | Saving strategy (e.g., `"no"`, `"epoch"`, `"steps"`)                     | `"no"`                                     |
| `--learning_rate`                | Initial learning rate for training                                       | `3e-4`                                     |
| `--warmup_ratio`                 | Ratio of total steps used for learning rate warmup                      | `0.03`                                     |
| `--lr_scheduler_type`            | Learning rate scheduler type (e.g., `"cosine"`, `"linear"`)              | `"cosine"`                                 |
| `--max_grad_norm`                | Maximum gradient norm for gradient clipping                              | `0.3`                                      |
| `--logging_steps`                | Interval (in steps) for logging training progress                        | `1`                                        |
| `--do_train`                     | Whether to perform training                                             | Enabled                                    |
| `--do_eval`                      | Whether to perform evaluation                                           | Enabled                                    |
| `--use_habana`                   | Enable Habana Gaudi training                                            | Enabled                                    |
| `--use_lazy_mode`                | Use Habana Lazy Mode for optimization                                   | Enabled                                    |
| `--throughput_warmup_steps`      | Number of warmup steps for measuring throughput                         | `3`                                        |
| `--use_peft`                     | Use Parameter-Efficient Fine-Tuning (PEFT)                              | `True`                                     |
| `--lora_r`                       | LoRA rank for low-rank adaptation                                       | `4`                                        |
| `--lora_alpha`                   | LoRA scaling factor                                                     | `16`                                       |
| `--lora_dropout`                 | Dropout probability for LoRA layers                                     | `0.05`                                     |
| `--lora_target_modules`          | Target modules for applying LoRA                                        | `"q_proj", "v_proj", "k_proj", "o_proj"`      |
| `--max_seq_length`               | Maximum sequence length for training                                   | `512`                                      |
| `--adam_epsilon`                 | Epsilon value for Adam optimizer                                       | `1e-08`                                    |
| `--use_flash_attention`          | Enable Flash Attention optimization                                    | Enabled                                    |
