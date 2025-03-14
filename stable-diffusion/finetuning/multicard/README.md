# Text to Image Generation Finetuning

This directory contains scripts that showcase how to perform training/fine-tuning of Stable Diffusion models on Habana Gaudi using Multi Card.

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

Install DeepSpeed 

```sh
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```



To download example training datasets locally, run:

```sh
python optimum-habana/examples/stable-diffusion/training/download_train_datasets.py
```


## Fine-Tuning for Stable Diffusion XL

The `train_text_to_image_sdxl.py` script shows how to implement the fine-tuning of Stable Diffusion XL models on Gaudi.

To train Stable Diffusion XL on a multi-card Gaudi system, use:

```sh
PT_HPU_RECIPE_CACHE_CONFIG=/tmp/stdxl_recipe_cache,True,1024  \
python optimum-habana/examples/gaudi_spawn.py --world_size 8 --use_mpi optimum-habana/examples/stable-diffusion/training/train_text_to_image_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --dataset_name lambdalabs/naruto-blip-captions \
    --resolution 512 \
    --crop_resolution 512 \
    --center_crop \
    --random_flip \
    --proportion_empty_prompts=0.2 \
    --train_batch_size 16 \
    --max_train_steps 336 \
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
    --checkpointing_steps 336 \
    --mediapipe dataset_sdxl_mediapipe \
    --adjust_throughput
```

### Note 
1. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 
2. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example

### variables 

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--world_size`                       | Number of cards to utilize                 | 8                                 |
| `--use_mpi`                          | Use MPI for distributed execution          |                                   |
| `--pretrained_model_name_or_path`    | Path/name of pretrained SDXL model         | `stabilityai/stable-diffusion-xl-base-1.0` |
| `--pretrained_vae_model_name_or_path`| Path/name of pretrained VAE model          | `madebyollin/sdxl-vae-fp16-fix`  |
| `--dataset_name`                     | Name of the training dataset               | `lambdalabs/naruto-blip-captions` |
| `--resolution`                       | Resolution of input images                 | 512                               |
| `--crop_resolution`                  | Resolution for cropped images              | 512                               |
| `--center_crop`                      | Enable center cropping of images           |                                   |
| `--random_flip`                      | Enable random flipping of images           |                                   |
| `--proportion_empty_prompts`         | Fraction of prompts to leave empty         | 0.2                               |
| `--train_batch_size`                 | Batch size for training                    | 16                                |
| `--max_train_steps`                  | Maximum number of training steps           | 336                               |
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
| `--checkpointing_steps`              | Steps between model checkpoints            | 336                               |
| `--mediapipe`                        | Mediapipe dataset or config                | `dataset_sdxl_mediapipe`          |
| `--adjust_throughput`                | Adjust settings for throughput             |                                   |

## DreamBooth

DreamBooth is a technique for personalizing text-to-image models like Stable Diffusion using only a few images (typically 3-5) of a specific subject. The `train_dreambooth.py` script demonstrates how to implement this training process and adapt it for Stable Diffusion.


### Full Model Fine-Tuning
To launch the multi-card Stable Diffusion training, use:

```sh
python optimum-habana/examples/gaudi_spawn.py --world_size 8 --use_mpi optimum-habana/examples/stable-diffusion/training/train_dreambooth.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="dog" \
    --output_dir="dog_sd" \
    --class_data_dir="path-to-class-images" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks dog" \
    --class_prompt="a photo of dog" \
    --resolution=512 \
    --train_batch_size=1 \
    --num_class_images=200 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=800 \
    --mixed_precision=bf16 \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/stable-diffusion \
    full
```

### Note 
1. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 
2. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example


### Variables 

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--world_size`                       | Number of cards to utilize                 | 8                                 |
| `--use_mpi`                          | Use MPI for distributed execution          |                                   |
| `--pretrained_model_name_or_path`    | Path/name of pretrained model              | `"CompVis/stable-diffusion-v1-4"`|
| `--instance_data_dir`                | Directory with instance data               | `"dog"`                           |
| `--output_dir`                       | Directory to save model outputs            | `"dog_sd"`                        |
| `--class_data_dir`                   | Directory with class images                | `"path-to-class-images"`          |
| `--with_prior_preservation`          | Enable prior preservation                  |                                   |
| `--prior_loss_weight`                | Weight for prior preservation loss         | 1.0                               |
| `--instance_prompt`                  | Prompt for instance data                   | `"a photo of sks dog"`            |
| `--class_prompt`                     | Prompt for class data                      | `"a photo of dog"`                |
| `--resolution`                       | Resolution of input images                 | 512                               |
| `--train_batch_size`                 | Batch size for training                    | 1                                 |
| `--num_class_images`                 | Number of class images to generate         | 200                               |
| `--gradient_accumulation_steps`      | Steps for gradient accumulation            | 1                                 |
| `--learning_rate`                    | Learning rate for optimization             | 5e-6                              |
| `--lr_scheduler`                     | Learning rate scheduler type               | `"constant"`                      |
| `--lr_warmup_steps`                  | Number of warmup steps for LR              | 0                                 |
| `--max_train_steps`                  | Maximum number of training steps           | 800                               |
| `--mixed_precision`                  | Precision type for mixed training          | `bf16`                            |
| `--use_hpu_graphs_for_training`      | Use HPU graphs for training optimization   |                                   |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--gaudi_config_name`                | Gaudi configuration name                   | `Habana/stable-diffusion`         |
| `full`                               | Unclear argument (possibly a typo)         | N/A                               |

### PEFT Model Fine-Tuning
We provide DreamBooth examples demonstrating how to use LoRA, LoKR, LoHA, and OFT adapters to fine-tune the UNet or text encoder.

To run the multi-card training, use:

```sh
python optimum-habana/examples/gaudi_spawn.py --world_size 8 --use_mpi optimum-habana/examples/stable-diffusion/training/train_dreambooth.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
    --instance_data_dir="dog" \
    --output_dir="dog_sd" \
    --class_data_dir="path-to-class-images" \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks dog" \
    --class_prompt="a photo of dog" \
    --resolution=512 \
    --train_batch_size=1 \
    --num_class_images=200 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=800 \
    --mixed_precision=bf16 \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --gaudi_config_name Habana/stable-diffusion \
    lora --unet_r 8 --unet_alpha 8
```

### Note
1. When using PEFT method we can use a much higher learning rate compared to vanilla dreambooth. Here we use `1e-4` instead of the usual `5e-6`

2. You could check each adapter's specific arguments with --help, for example:
```sh
python3 optimum-habana/examples/stable-diffusion/training/train_dreambooth.py oft --help
```
3. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 

4. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example


### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--world_size`                       | Number of cards to utilize                 | 8                                 |
| `--use_mpi`                          | Use MPI for distributed execution          |                                   |
| `--pretrained_model_name_or_path`    | Path/name of pretrained model              | `"CompVis/stable-diffusion-v1-4"`|
| `--instance_data_dir`                | Directory with instance data               | `"dog"`                           |
| `--output_dir`                       | Directory to save model outputs            | `"dog_sd"`                        |
| `--class_data_dir`                   | Directory with class images                | `"path-to-class-images"`          |
| `--with_prior_preservation`          | Enable prior preservation                  |                                   |
| `--prior_loss_weight`                | Weight for prior preservation loss         | 1.0                               |
| `--instance_prompt`                  | Prompt for instance data                   | `"a photo of sks dog"`            |
| `--class_prompt`                     | Prompt for class data                      | `"a photo of dog"`                |
| `--resolution`                       | Resolution of input images                 | 512                               |
| `--train_batch_size`                 | Batch size for training                    | 1                                 |
| `--num_class_images`                 | Number of class images to generate         | 200                               |
| `--gradient_accumulation_steps`      | Steps for gradient accumulation            | 1                                 |
| `--learning_rate`                    | Learning rate for optimization             | 1e-4                              |
| `--lr_scheduler`                     | Learning rate scheduler type               | `"constant"`                      |
| `--lr_warmup_steps`                  | Number of warmup steps for LR              | 0                                 |
| `--max_train_steps`                  | Maximum number of training steps           | 800                               |
| `--mixed_precision`                  | Precision type for mixed training          | `bf16`                            |
| `--use_hpu_graphs_for_training`      | Use HPU graphs for training optimization   |                                   |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--gaudi_config_name`                | Gaudi configuration name                   | `Habana/stable-diffusion`         |
| `lora`                               | Enable LoRA (possibly a flag or typo)      | N/A                               |
| `--unet_r`                           | LoRA rank for UNet                         | 8                                 |
| `--unet_alpha`                       | LoRA alpha value for UNet                  | 8                                 |



### DreamBooth LoRA Fine-Tuning with Stable Diffusion XL

To launch Stable Diffusion XL LoRA training on a Multi-card Gaudi system, use:

```sh
python optimum-habana/examples/gaudi_spawn.py --world_size 8 --use_mpi optimum-habana/examples/stable-diffusion/training/train_dreambooth_lora_sdxl.py \
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
### Note
1. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 
2. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example 


### Variables 
| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--world_size`                       | Number of cards to utilize                 | 8                                 |
| `--use_mpi`                          | Use MPI for distributed execution          |                                   |
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

To launch FLUX.1-dev LoRA training on a multi Gaudi card, use:

```sh
python optimum-habana/examples/gaudi_spawn.py --world_size 8 --use_mpi optimum-habana/examples/stable-diffusion/training/train_dreambooth_lora_flux.py \
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

### Note 
1. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 
2. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example 

### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--world_size`                       | Number of cards to utilize                 | 8                                 |
| `--use_mpi`                          | Use MPI for distributed execution          |                                   |
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

## ControlNet Training

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala. It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image. This example is adapted from [controlnet example in the diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/controlnet#training).

To download the example conditioning images locally, run:

```sh
python optimum-habana/examples/stable-diffusion/training/download_train_datasets.py
```

You can run these fine-tuning scripts in a distributed fashion as follows:

```sh
python optimum-habana/examples/gaudi_spawn.py --use_mpi --world_size 8 optimum-habana/examples/stable-diffusion/training/train_controlnet.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --output_dir=/tmp/stable_diffusion1_4 \
    --dataset_name=fusing/fill50k \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "./cnet/conditioning_image_1.png" "./cnet/conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=4 \
    --throughput_warmup_steps 3 \
    --use_hpu_graphs \
    --sdp_on_bf16 \
    --bf16 \
    --trust_remote_code
```
### Note 
1. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 
2. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example


### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--use_mpi`                          | Use MPI for distributed execution          |                                   |
| `--world_size`                       | Number of cards to utilize                 | 8                                 |
| `--pretrained_model_name_or_path`    | Path/name of pretrained model              | `CompVis/stable-diffusion-v1-4`  |
| `--output_dir`                       | Directory to save model outputs            | `/tmp/stable_diffusion1_4`        |
| `--dataset_name`                     | Name of the training dataset               | `fusing/fill50k`                  |
| `--resolution`                       | Resolution of input images                 | 512                               |
| `--learning_rate`                    | Learning rate for optimization             | 1e-5                              |
| `--validation_image`                 | Paths to validation conditioning images    | `"./cnet/conditioning_image_1.png" "./cnet/conditioning_image_2.png"` |
| `--validation_prompt`                | Prompts for validation images              | `"red circle with blue background" "cyan circle with brown floral background"` |
| `--train_batch_size`                 | Batch size for training                    | 4                                 |
| `--throughput_warmup_steps`          | Steps for throughput warmup                | 3                                 |
| `--use_hpu_graphs`                   | Use HPU graphs for optimization            |                                   |
| `--sdp_on_bf16`                      | Use SDP with BF16 precision                |                                   |
| `--bf16`                             | Use BF16 precision for computation        |                                   |
| `--trust_remote_code`                | Allow execution of remote code             |                                   |