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


## DreamBooth

DreamBooth is a technique for personalizing text-to-image models like Stable Diffusion using only a few images (typically 3-5) of a specific subject. The `train_dreambooth.py` script demonstrates how to implement this training process and adapt it for Stable Diffusion.

### DreamBooth LoRA Fine-Tuning with Stable Diffusion XL

To launch Stable Diffusion XL LoRA training on a single-card Gaudi system, use:

```sh
python optimum-habana/examples/stable-diffusion/training/train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
    --instance_data_dir="dog" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --output_dir="lora-trained-xl" \
    --mixed_precision="bf16" \
    --instance_prompt="a photo of sks dog" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --validation_prompt="A photo of sks dog in a bucket" \
    --validation_epochs=25 \
    --seed=0 \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --gaudi_config_name Habana/stable-diffusion

```

### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--pretrained_model_name_or_path`    | Path/name of pretrained SDXL model         | `"stabilityai/stable-diffusion-xl-base-1.0"` |
| `--instance_data_dir`                | Directory with instance data               | `"dog"`                           |
| `--pretrained_vae_model_name_or_path`| Path/name of pretrained VAE model          | `"madebyollin/sdxl-vae-fp16-fix"`|
| `--output_dir`                       | Directory to save model outputs            | `"lora-trained-xl"`               |
| `--mixed_precision`                  | Precision type for mixed training          | `"bf16"`                          |
| `--instance_prompt`                  | Prompt for instance data                   | `"a photo of sks dog"`            |
| `--resolution`                       | Resolution of input images                 | 1024                              |
| `--train_batch_size`                 | Batch size for training                    | 1                                 |
| `--gradient_accumulation_steps`      | Steps for gradient accumulation            | 4                                 |
| `--learning_rate`                    | Learning rate for optimization             | 1e-4                              |
| `--lr_scheduler`                     | Learning rate scheduler type               | `"constant"`                      |
| `--lr_warmup_steps`                  | Number of warmup steps for LR              | 0                                 |
| `--max_train_steps`                  | Maximum number of training steps           | 500                               |
| `--validation_prompt`                | Prompt for validation images               | `"A photo of sks dog in a bucket"`|
| `--validation_epochs`                | Epochs between validation runs             | 25                                |
| `--seed`                             | Random seed for reproducibility           | 0                                 |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--use_hpu_graphs_for_training`      | Use HPU graphs for training optimization   |                                   |
| `--gaudi_config_name`                | Gaudi configuration name                   | `Habana/stable-diffusion`         |


### DreamBooth LoRA Fine-Tuning with FLUX.1-dev

Before running FLUX.1-dev model, you need to:

1. Agree to the Terms and Conditions for using FLUX.1-dev model at `https://huggingface.co/black-forest-labs/FLUX.1-dev`

2. Authenticate with HuggingFace using your HF Token. For authentication, run:

```sh
huggingface-cli login
```

To launch FLUX.1-dev LoRA training on a single Gaudi card, use:

```sh
python optimum-habana/examples/stable-diffusion/training/train_dreambooth_lora_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --dataset="dog" \
    --prompt="a photo of sks dog" \
    --output_dir="dog_lora_flux" \
    --mixed_precision="bf16" \
    --weighting_scheme="none" \
    --resolution=1024 \
    --train_batch_size=1 \
    --learning_rate=1e-4 \
    --guidance_scale=1 \
    --report_to="tensorboard" \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --cache_latents \
    --rank=4 \
    --max_train_steps=500 \
    --seed="0" \
    --use_hpu_graphs_for_inference \
    --use_hpu_graphs_for_training \
    --gaudi_config_name="Habana/stable-diffusion"
```

### variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `train_dreambooth_lora_flux.py`      | Script to run (not a flag)                 | N/A                               |
| `--pretrained_model_name_or_path`    | Path/name of pretrained FLUX model         | `"black-forest-labs/FLUX.1-dev"` |
| `--dataset`                          | Dataset or directory for training          | `"dog"`                           |
| `--prompt`                           | Prompt for training images                 | `"a photo of sks dog"`            |
| `--output_dir`                       | Directory to save model outputs            | `"dog_lora_flux"`                 |
| `--mixed_precision`                  | Precision type for mixed training          | `"bf16"`                          |
| `--weighting_scheme`                 | Scheme for weighting loss                  | `"none"`                          |
| `--resolution`                       | Resolution of input images                 | 1024                              |
| `--train_batch_size`                 | Batch size for training                    | 1                                 |
| `--learning_rate`                    | Learning rate for optimization             | 1e-4                              |
| `--guidance_scale`                   | Scale for classifier-free guidance         | 1                                 |
| `--report_to`                        | Tool for reporting metrics                 | `"tensorboard"`                   |
| `--gradient_accumulation_steps`      | Steps for gradient accumulation            | 4                                 |
| `--gradient_checkpointing`           | Enable gradient checkpointing              |                                   |
| `--lr_scheduler`                     | Learning rate scheduler type               | `"constant"`                      |
| `--lr_warmup_steps`                  | Number of warmup steps for LR              | 0                                 |
| `--cache_latents`                    | Cache latent representations               |                                   |
| `--rank`                             | Rank for LoRA adaptation                   | 4                                 |
| `--max_train_steps`                  | Maximum number of training steps           | 500                               |
| `--seed`                             | Random seed for reproducibility           | `"0"`                             |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--use_hpu_graphs_for_training`      | Use HPU graphs for training optimization   |                                   |
| `--gaudi_config_name`                | Gaudi configuration name                   | `"Habana/stable-diffusion"`       |

