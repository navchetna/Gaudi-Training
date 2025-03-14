# Text to Image Generation 

## Requirements

Install Optimum Habana

```sh
git clone https://github.com/huggingface/optimum-habana.git

pip install ./optimum-habana
```
Install text to image generation specific requirements

```sh
pip install -r examples/T2I/requirements.txt
```

Install DeepSpeed 

```sh
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```

## Stable Diffusion

Stable Diffusion is a deep learning-based text-to-image model that generates high-quality images from text prompts.


Run the following command to generate images from text prompts using Stable Diffusion:

```sh
python optimum-habana/examples/gaudi_spawn.py \
    --world_size 8 optimum-habana/examples/stable-diffusion/training/text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 20 \
    --batch_size 4 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --distributed

```

### Note

1. You can send one or more prompts at once using the `--prompts` parameter. The example above uses two prompts, but you can specify any number of prompts as needed.
2. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 

### Variables 
| Variable                  | Short Description                        | Default Value         |
|---------------------------|------------------------------------------|-----------------------|
| `--world_size`            | Number of processes for distributed run  | 8                     |
| `--model_name_or_path`    | Path or name of the model                | `CompVis/stable-diffusion-v1-4` |
| `--prompts`               | Text prompts for image generation        | "A cat holding a sign that says hello world"                  |
| `--num_images_per_prompt` | Number of images per prompt              | 20                     |
| `--batch_size`            | Number of images processed at once       | 4                     |
| `--image_save_dir`        | Directory to save generated images       | `./outputs`           |
| `--use_habana`            | Enable Habana hardware support           |                  |
| `--use_hpu_graphs`        | Use HPU graphs for optimization          |                  |
| `--gaudi_config`          | Gaudi configuration file or preset       | `Habana/stable-diffusion` |
| `--sdp_on_bf16`           | Use SDP with BF16 precision              |                  |
| `--bf16`                  | Use BF16 precision for computation      |                  |
| `--distributed`           | Enable distributed training/inference    |                  | 


## Latent Diffusion Model for 3D (LDM3D)

LDM3D generates both image and depth map data from a given text prompt, allowing users to generate RGBD images from text prompts.


Run the following command to generate images from text prompts using Latent Diffusion Model:

```sh
python optimum-habana/examples/gaudi_spawn.py --world_size 2 optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path "Intel/ldm3d-4c" \
    --prompts "An image of a squirrel in Picasso style" "A shiny flying horse taking off" \
    --num_images_per_prompt 10 \
    --batch_size 2 \
    --height 768 \
    --width 768 \
    --image_save_dir /tmp/stable_diffusion_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion-2 \
    --ldm3d \
    --distributed

```

### Note
1. You can send one or more prompts at once using the `--prompts` parameter. The example above uses two prompts, but you can specify any number of prompts as needed.

2. Use `--ldm3d` parameter along with LDM supported model in order to create `RGBD` images 

3. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 

### Variables

| Variable                  | Short Description                        | Default Value         |
|---------------------------|------------------------------------------|-----------------------|
| `--world_size`            | Number of processes for distributed run  | 2                     |
| `--model_name_or_path`    | Path or name of the model                | `Intel/ldm3d-4c`      |
| `--prompts`               | Text prompts for image generation        | "An image of a squirrel in Picasso style"                  |
| `--num_images_per_prompt` | Number of images per prompt              | 10                     |
| `--batch_size`            | Number of images processed at once       | 2                     |
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
python optimum-habana/examples/gaudi_spawn.py --world_size 2 optimum-habana/examples/stable-diffusion/text_to_image_generation.py \
    --model_name_or_path CompVis/stable-diffusion-v1-4 \
    --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
    --prompts "futuristic-looking woman" "a rusty robot" \
    --control_image https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png \
    --num_images_per_prompt 16 \
    --batch_size 4 \
    --image_save_dir /tmp/controlnet_images \
    --use_habana \
    --use_hpu_graphs \
    --gaudi_config Habana/stable-diffusion \
    --sdp_on_bf16 \
    --bf16 \
    --distributed

```

### Note
1. You can send 1 or more prompts at once using parameter `--prompts` need not be two prompts as mentioned above in example. 
2. Use `--contrl_image` to set the control image that is required by an ControlNet Model  
3. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 


### Variables

| Variable                         | Short Description                        | Default Value         |
|----------------------------------|------------------------------------------|-----------------------|
| `--world_size`            | Number of processes for distributed run  | 2                     |
| `--model_name_or_path`           | Path or name of the base model           | `CompVis/stable-diffusion-v1-4` |
| `--controlnet_model_name_or_path`| Path or name of the ControlNet model     | lllyasviel/sd-controlnet-canny                  |
| `--prompts`                      | Text prompts for image generation        | "An image of a squirrel in Picasso style"                  |
| `--control_image`                | URL or path to control image             | https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png                  |
| `--num_images_per_prompt`        | Number of images per prompt              | 16                     |
| `--batch_size`                   | Number of images processed at once       | 4                     |
| `--image_save_dir`               | Directory to save generated images       | `./outputs`           |
| `--use_habana`                   | Enable Habana hardware support           |                  |
| `--use_hpu_graphs`               | Use HPU graphs for optimization          |                  |
| `--gaudi_config`                 | Gaudi configuration file or preset       | Habana/stable-diffusion        |
| `--sdp_on_bf16`                  | Use SDP with BF16 precision              |                  |
| `--bf16`                         | Use BF16 precision for computation      |                  |
