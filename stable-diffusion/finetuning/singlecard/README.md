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

### variables 

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


## Textual Inversion

[Textual Inversion](https://arxiv.org/abs/2208.01618) is a method to personalize text2image models like Stable Diffusion on your own images using just 3-5 examples.

The textual_inversion.py script shows how to implement the training procedure on Habana Gaudi.

In the examples below, we will use a set of cat images from the following dataset: [https://huggingface.co/datasets/diffusers/cat_toy_example](https://huggingface.co/datasets/diffusers/cat_toy_example)

To download this and other example training datasets locally, run:

```sh
python optimum-habana/examples/stable-diffusion/training/download_train_datasets.py
```

launch training using 

```sh
python optimum-habana/examples/stable-diffusion/training/textual_inversion.py \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
    --train_data_dir ./cat \
    --learnable_property object \
    --placeholder_token "<cat-toy>" \
    --initializer_token toy \
    --resolution 512 \
    --train_batch_size 4 \
    --max_train_steps 3000 \
    --learning_rate 5.0e-04 \
    --scale_lr \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --output_dir /tmp/textual_inversion_cat \
    --save_as_full_pipeline \
    --gaudi_config_name Habana/stable-diffusion \
    --throughput_warmup_steps 3
```

### Note 
1. Change --resolution to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.

2. As described in [the official paper](https://arxiv.org/abs/2208.01618), only one embedding vector is used for the placeholder token, e.g. `"<cat-toy>"`. However, one can also add multiple embedding vectors for the placeholder token to increase the number of fine-tuneable parameters. This can help the model to learn more complex details. To use multiple embedding vectors, you can define `--num_vectors` to a number larger than one, e.g.: `--num_vectors 5`. The saved textual inversion vectors will then be larger in size compared to the default case.

### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `textual_inversion.py`               | Script to run (not a flag)                 | N/A                               |
| `--pretrained_model_name_or_path`    | Path/name of pretrained model              | `CompVis/stable-diffusion-v1-4`  |
| `--train_data_dir`                   | Directory with training data               | `./cat`                           |
| `--learnable_property`               | Property to learn (e.g., object, style)    | `object`                          |
| `--placeholder_token`                | Token to represent learned concept         | `"<cat-toy>"`                    |
| `--initializer_token`                | Initial token for embedding                | `toy`                             |
| `--resolution`                       | Resolution of input images                 | 512                               |
| `--train_batch_size`                 | Batch size for training                    | 4                                 |
| `--max_train_steps`                  | Maximum number of training steps           | 3000                              |
| `--learning_rate`                    | Learning rate for optimization             | 5.0e-04                           |
| `--scale_lr`                         | Scale learning rate by batch size          |                                   |
| `--lr_scheduler`                     | Learning rate scheduler type               | `constant`                        |
| `--lr_warmup_steps`                  | Number of warmup steps for LR              | 0                                 |
| `--output_dir`                       | Directory to save model outputs            | `/tmp/textual_inversion_cat`      |
| `--save_as_full_pipeline`            | Save model as a full pipeline              |                                   |
| `--gaudi_config_name`                | Gaudi configuration name                   | `Habana/stable-diffusion`         |
| `--throughput_warmup_steps`          | Steps for throughput warmup                | 3                                 |


## ControlNet Training

ControlNet was introduced in [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) by Lvmin Zhang and Maneesh Agrawala. It is a type of model for controlling StableDiffusion by conditioning the model with an additional input image. This example is adapted from [controlnet example in the diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/controlnet#training).

To download the example conditioning images locally, run:

```sh
python optimum-habana/examples/stable-diffusion/training/download_train_datasets.py
```

Then Proceed to Training with command 

```sh
python optimum-habana/examples/stable-diffusion/training/train_controlnet.py \
   --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4\
   --output_dir=/tmp/stable_diffusion1_4 \
   --dataset_name=fusing/fill50k \
   --resolution=512 \
   --learning_rate=1e-5 \
   --validation_image "./cnet/conditioning_image_1.png" "./cnet/conditioning_image_2.png" \
   --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
   --train_batch_size=4 \
   --throughput_warmup_steps=3 \
   --use_hpu_graphs \
   --sdp_on_bf16 \
   --bf16 \
   --trust_remote_code
```

### variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `train_controlnet.py`                | Script to run (not a flag)                 | N/A                               |
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






