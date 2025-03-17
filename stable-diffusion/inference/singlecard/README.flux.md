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
---