# Text to Image Generation Finetuning

This directory contains scripts that showcase how to perform training/fine-tuning of Stable Diffusion models on Habana Gaudi using Single Card.

## Install Requirements 

Install Optimum Habana

```sh
git clone https://github.com/huggingface/optimum-habana.git

pip install ./optimum-habana
```
Install additional task specific requirements

```sh
pip install -r optimum-habana/examples/stable-diffusion/training/requirements.txt
```


## Fine-Tuning for Stable Diffusion XL

The `train_text_to_image_sdxl.py` script shows how to implement the fine-tuning of Stable Diffusion XL models on Gaudi.



To train Stable Diffusion XL on a single Gaudi card, use:

```sh
python optimum-habana/examples/stable-diffusion/training/train_text_to_image_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --dataset_name lambdalabs/naruto-blip-captions \
    --resolution 512 \
    --crop_resolution 512 \
    --center_crop \
    --random_flip \
    --proportion_empty_prompts=0.2 \
    --train_batch_size 16 \
    --max_train_steps 2500 \
    --learning_rate 1e-05 \
    --max_grad_norm 1 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --output_dir sdxl_model_output \
    --gaudi_config_name Habana/stable-diffusion \
    --throughput_warmup_steps 3 \
    --dataloader_num_workers 8 \
    --sdp_on_bf16 \
    --bf16 \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --validation_prompt="a cute naruto creature" \
    --validation_epochs 48 \
    --checkpointing_steps 2500 \
    --logging_step 10 \
    --adjust_throughput
```

### Variables 

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `train_text_to_image_sdxl.py`        | Script to run (not a flag)                 | N/A                               |
| `--pretrained_model_name_or_path`    | Path/name of pretrained SDXL model         | `stabilityai/stable-diffusion-xl-base-1.0` |
| `--pretrained_vae_model_name_or_path`| Path/name of pretrained VAE model          | `madebyollin/sdxl-vae-fp16-fix`  |
| `--dataset_name`                     | Name of the training dataset               | `lambdalabs/naruto-blip-captions` |
| `--resolution`                       | Resolution of input images                 | 512                               |
| `--crop_resolution`                  | Resolution for cropped images              | 512                               |
| `--center_crop`                      | Enable center cropping of images           |                                   |
| `--random_flip`                      | Enable random flipping of images           |                                   |
| `--proportion_empty_prompts`         | Fraction of prompts to leave empty         | 0.2                               |
| `--train_batch_size`                 | Batch size for training                    | 16                                |
| `--max_train_steps`                  | Maximum number of training steps           | 2500                              |
| `--learning_rate`                    | Learning rate for optimization             | 1e-05                             |
| `--max_grad_norm`                    | Maximum gradient norm for clipping         | 1                                 |
| `--lr_scheduler`                     | Learning rate scheduler type               | `constant`                        |
| `--lr_warmup_steps`                  | Number of warmup steps for LR              | 0                                 |
| `--output_dir`                       | Directory to save model outputs            | `sdxl_model_output`               |
| `--gaudi_config_name`                | Gaudi configuration name                   | `Habana/stable-diffusion`         |
| `--throughput_warmup_steps`          | Steps for throughput warmup                | 3                                 |
| `--dataloader_num_workers`           | Number of workers for data loading         | 8                                 |
| `--sdp_on_bf16`                      | Use SDP with BF16 precision                |                                   |
| `--bf16`                             | Use BF16 precision for computation        |                                   |
| `--use_hpu_graphs_for_training`      | Use HPU graphs for training optimization   |                                   |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--validation_prompt`                | Prompt for validation images               | `"a cute naruto creature"`        |
| `--validation_epochs`                | Epochs between validation runs             | 48                                |
| `--checkpointing_steps`              | Steps between model checkpoints            | 2500                              |
| `--logging_step`                     | Steps between logging updates              | 10                                |
| `--adjust_throughput`                | Adjust settings for throughput             |                                   |

