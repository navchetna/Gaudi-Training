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
---
