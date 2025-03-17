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

