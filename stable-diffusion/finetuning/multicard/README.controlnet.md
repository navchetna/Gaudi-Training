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