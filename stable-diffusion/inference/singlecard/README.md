# Text to Image Generation 

## Requirements

Install Optimum Habana

```sh
git clone https://github.com/huggingface/optimum-habana.git

pip install ./optimum-habana
```
Install additional task specific requirements

```sh
pip install -r optimum-habana/examples/stable-diffusion/requirements.txt
```

## Stable Diffusion

Stable Diffusion is a deep learning-based text-to-image model that generates high-quality images from text prompts.


Run the following command to generate images from text prompts using Stable Diffusion:

```sh
python optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 32 \
    --batch_size 8 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Note

1. You can send one or more prompts at once using the `--prompts` parameter. The example above uses two prompts, but you can specify any number of prompts as needed.

### Variables

| Variable                  | Short Description                        | Default Value         |
|---------------------------|------------------------------------------|-----------------------|
| `--model_name_or_path`    | Path or name of the model                | `CompVis/stable-diffusion-v1-4` |
| `--prompts`               | Text prompts for image generation        | "A cat holding a sign that says hello world"                  |
| `--num_images_per_prompt` | Number of images per prompt              | 32                     |
| `--batch_size`            | Number of images processed at once       | 8                     |
| `--image_save_dir`        | Directory to save generated images       | `./outputs`           |
| `--use_habana`            | Enable Habana hardware support           |                  |
| `--use_hpu_graphs`        | Use HPU graphs for optimization          |                  |
| `--gaudi_config`          | Gaudi configuration file or preset       | Habana/stable-diffusion         |
| `--sdp_on_bf16`           | Use SDP with BF16 precision              |                  |
| `--bf16`                  | Use BF16 precision for computation      |                  |


## Latent Diffusion Model for 3D (LDM3D)

LDM3D generates both image and depth map data from a given text prompt, allowing users to generate RGBD images from text prompts.


Run the following command to generate images from text prompts using Latent Diffusion Model:

```sh
python optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path "Intel/ldm3d-4c" \
    --prompts "An image of a squirrel in Picasso style" \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --ldm3d
```

### Note
1. You can send one or more prompts at once using the `--prompts` parameter. The example above uses two prompts, but you can specify any number of prompts as needed.

2. Use `--ldm3d` parameter along with LDM supported model in order to create `RGBD` images 


### Variables

| Variable                  | Short Description                        | Default Value         |
|---------------------------|------------------------------------------|-----------------------|
| `--model_name_or_path`    | Path or name of the model                | `Intel/ldm3d-4c`      |
| `--prompts`               | Text prompts for image generation        | "An image of a squirrel in Picasso style"                  |
| `--num_images_per_prompt` | Number of images per prompt              | 28                     |
| `--batch_size`            | Number of images processed at once       | 7                    |
| `--height`                | Height of generated images (pixels)      | 512                   |
| `--width`                 | Width of generated images (pixels)       | 512                   |
| `--image_save_dir`        | Directory to save generated images       | `./outputs`           |
| `--use_habana`            | Enable Habana hardware support           |                  |
| `--use_hpu_graphs`        | Use HPU graphs for optimization          |                  |
| `--gaudi_config`          | Gaudi configuration file or preset       | Habana/stable-diffusion         |
| `--ldm3d`                 | Enable LDM3D-specific features           |                  |


##

## ControlNet

ControlNet was introduced in Adding Conditional Control to Text-to-Image Diffusion Models by conditioning the Stable Diffusion model with an additional input image. This allows for precise control over the composition of generated images using various features such as edges, pose, depth, and more.

Run the following command to generate images from text prompts using ControlNet:

```sh
python optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 28 \
    --batch_size 7 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Note
1. You can send 1 or more prompts at once using parameter `--prompts` need not be two prompts as mentioned above in example. 
2. Use `--contrl_image` to set the control image that is required by an ControlNet Model  


### Variables

| Variable                         | Short Description                        | Default Value         |
|----------------------------------|------------------------------------------|-----------------------|
| `--model_name_or_path`           | Path or name of the base model           | `CompVis/stable-diffusion-v1-4` |
| `--controlnet_model_name_or_path`| Path or name of the ControlNet model     | lllyasviel/sd-controlnet-canny                  |
| `--prompts`                      | Text prompts for image generation        | "An image of a squirrel in Picasso style"                  |
| `--control_image`                | URL or path to control image             | https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png                  |
| `--num_images_per_prompt`        | Number of images per prompt              | 28                     |
| `--batch_size`                   | Number of images processed at once       | 7                     |
| `--image_save_dir`               | Directory to save generated images       | `./outputs`           |
| `--use_habana`                   | Enable Habana hardware support           |                  |
| `--use_hpu_graphs`               | Use HPU graphs for optimization          |                  |
| `--gaudi_config`                 | Gaudi configuration file or preset       | Habana/stable-diffusion        |
| `--sdp_on_bf16`                  | Use SDP with BF16 precision              |                  |
| `--bf16`                         | Use BF16 precision for computation      |                  |


## Flux .1

Flux text-to-image generation is an AI-based technique that dynamically refines images from text prompts using diffusion models or neural fields. It leverages continuous transformations in latent space to produce high-quality, coherent visuals with adaptive style control.

Run the following command to generate images from text prompts using Flux .1:

### For non dev model 

```sh
python optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path black-forest-labs/FLUX.1-schnell \
    --prompts "A cat holding a sign that says hello world" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 4 \
    --image_save_dir /tmp/flux_1_images \
    --scheduler flow_match_euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### For dev model 

Before running FLUX.1-dev model, you need to:

1. Agree to the Terms and Conditions for using FLUX.1-dev model at `https://huggingface.co/black-forest-labs/FLUX.1-dev`

2. Authenticate with HuggingFace using your HF Token. For authentication, run:

```sh
huggingface-cli login
```
Run the following command to generate images from text prompts using Flux .1:

```sh
python optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path black-forest-labs/FLUX.1-dev \
    --prompts "A cat holding a sign that says hello world" \
    --num_images_per_prompt 10 \
    --batch_size 1 \
    --num_inference_steps 30 \
    --image_save_dir /tmp/flux_1_images \
    --scheduler flow_match_euler_discrete \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16
```

### Note

1. The parameter `--scheduler` is different for normal flux and flux-dev models 

2. You can send 1 or more prompts at once using parameter `--prompts` need not be two prompts as mentioned above in example.


### Variables

| Variable                  | Short Description                        | Default Value         |
|---------------------------|------------------------------------------|-----------------------|
| `--model_name_or_path`    | Path or name of the model                | `black-forest-labs/FLUX.1-schnell` |
| `--prompts`               | Text prompts for image generation        | "A cat holding a sign that says hello world"                  |
| `--num_images_per_prompt` | Number of images per prompt              | 10                   |
| `--batch_size`            | Number of images processed at once       | 1                     |
| `--num_inference_steps`   | Number of denoising steps                | 50 , 30                   |
| `--image_save_dir`        | Directory to save generated images       | `./outputs`           |
| `--scheduler`             | Scheduler algorithm for diffusion        | `euler` , `flow_match_euler_discrete` |
| `--use_habana`            | Enable Habana hardware support           |                  |
| `--use_hpu_graphs`        | Use HPU graphs for optimization          |                  |
| `--gaudi_config`          | Gaudi configuration file or preset       | Habana/stable-diffusion         |
| `--sdp_on_bf16`           | Use SDP with BF16 precision              |                  |
| `--bf16`                  | Use BF16 precision for computation      |                  |