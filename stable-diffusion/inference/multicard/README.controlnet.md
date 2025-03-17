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
---