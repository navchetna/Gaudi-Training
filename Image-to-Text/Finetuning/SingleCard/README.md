# Image to Text 

This directory contains a script that showcases how to finetune image to text generation on Intel® Gaudi® AI Accelerators using a single card.

## Requirements

### Run the Intel Gaudi Docker image:

```sh
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host"
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all $DOCKER_OPTS vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
cd root
```

### Install Optimum Habana

```sh
git clone https://github.com/huggingface/optimum-habana.git
pip install ./optimum-habana
```

### Install additional task specific requirements

```sh
pip install -r optimum-habana/examples/image-to-text/requirements.txt
```

## LoRA Finetune

Here are single-device command example

```sh
python3 optimum-habana/examples/image-to-text/run_image2text_lora_finetune.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset_name nielsr/docvqa_1200_examples \
    --bf16 True \
    --output_dir ./model_lora_llama \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --weight_decay 0.01 \
    --logging_steps 25 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 5e-5 \
    --warmup_steps  50 \
    --lr_scheduler_type "constant" \
    --input_column_names 'image' 'query' \
    --output_column_names 'answers' \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --lora_rank=8 \
    --lora_alpha=8 \
    --lora_dropout=0.1 \
    --low_cpu_mem_usage True \
    --max_seq_length=512 \
    --use_hpu_graphs_for_inference True \
    --lora_target_modules ".*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
```

### Note
1. The model **meta-llama/Llama-3.2-11B-Vision-Instruct** requires special access permissions. Please request access on the model's page on Hugging Face.

### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--model_name_or_path`               | Path/name of the model                     | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `--dataset_name`                     | Name of the training dataset               | `nielsr/docvqa_1200_examples`     |
| `--bf16`                             | Use BF16 precision for computation         | `True`                            |
| `--output_dir`                       | Directory to save model outputs            | `./model_lora_llama`              |
| `--num_train_epochs`                 | Number of training epochs                  | 2                                 |
| `--per_device_train_batch_size`      | Batch size per device for training         | 2                                 |
| `--per_device_eval_batch_size`       | Batch size per device for evaluation       | 2                                 |
| `--gradient_accumulation_steps`      | Steps for gradient accumulation            | 8                                 |
| `--weight_decay`                     | Weight decay for optimization              | 0.01                              |
| `--logging_steps`                    | Steps between logging updates              | 25                                |
| `--eval_strategy`                    | Evaluation strategy                        | `"no"`                            |
| `--save_strategy`                    | Model saving strategy                      | `"no"`                            |
| `--learning_rate`                    | Learning rate for optimization             | 5e-5                              |
| `--warmup_steps`                     | Number of warmup steps for LR              | 50                                |
| `--lr_scheduler_type`                | Learning rate scheduler type               | `"constant"`                      |
| `--input_column_names`               | Input column names in dataset              | `'image' 'query'`                 |
| `--output_column_names`              | Output column names in dataset             | `'answers'`                       |
| `--remove_unused_columns`            | Remove unused columns from dataset         | `False`                           |
| `--do_train`                         | Perform training                           |                                   |
| `--do_eval`                          | Perform evaluation                         |                                   |
| `--use_habana`                       | Enable Habana hardware support             |                                   |
| `--use_lazy_mode`                    | Use lazy mode for computation              |                                   |
| `--lora_rank`                        | Rank for LoRA adaptation                   | 8                                 |
| `--lora_alpha`                       | LoRA alpha value                           | 8                                 |
| `--lora_dropout`                     | Dropout rate for LoRA                      | 0.1                               |
| `--low_cpu_mem_usage`                | Reduce CPU memory usage                    | `True`                            |
| `--max_seq_length`                   | Maximum sequence length                    | 512                               |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  | `True`                            |
| `--lora_target_modules`              | LoRA target modules (regex)                | `".*(language_model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"` |