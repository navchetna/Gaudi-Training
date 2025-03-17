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

---