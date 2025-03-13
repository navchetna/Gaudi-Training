# Image to Text 

This directory contains a script that showcases how to perform image to text generation on Intel® Gaudi® AI Accelerators using multi cards.

## Requirements 

### Run the Intel Gaudi Docker image:

```sh
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host"
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all $DOCKER_OPTS vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
cd root
```

### Install Optimum Habana

```sh
git clone https://github.com/huggingface/optimum-habana.git
pip install ./optimum-habana
```

### Install additional task specific requirements

```sh
pip install -r optimum-habana/examples/image-to-text/requirements.txt
```

### Install DeepSpeed 

```sh
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```

## Multi card inference with BF16

To run inference 

```sh
PT_HPU_ENABLE_LAZY_COLLECTIVES=true python optimum-habana/examples/gaudi_spawn.py --use_deepspeed --world_size 2 optimum-habana/examples/image-to-text/run_pipeline.py \
    --model_name_or_path meta-llama/Llama-3.2-90B-Vision-Instruct \
    --image_path "https://llava-vl.github.io/static/images/view.jpg" \
    --use_hpu_graphs \
    --bf16 \
    --use_flash_attention \
    --flash_attention_recompute
```

### Note 
1. The model **meta-llama/Llama-3.2-90B-Vision-Instruct** requires special access permissions. Please request access on the model's page on Hugging Face.
2. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 
3. To use DeepSpeed instead of MPI, replace `--use_mpi` with `--use_deepspeed` in the previous example

### Variables 

| Variable Name                  | Short Explanation                        | Default Value                     |
|--------------------------------|------------------------------------------|-----------------------------------|
| `--use_deepspeed`              | Use DeepSpeed for optimization           |                                   |
| `--world_size`                 | Number of cards to utilize               | 2                                 |
| `run_pipeline.py`              | Pipeline script (not a flag)             | N/A                               |
| `--model_name_or_path`         | Path/name of the model                   | `meta-llama/Llama-3.2-90B-Vision-Instruct` |
| `--image_path`                 | URL or path to input image               | `"https://llava-vl.github.io/static/images/view.jpg"` |
| `--use_hpu_graphs`             | Use HPU graphs for optimization          |                                   |
| `--bf16`                       | Use BF16 precision for computation       |                                   |
| `--use_flash_attention`         | Enable flash attention mechanism        |                                   |
| `--flash_attention_recompute`  | Recompute flash attention                |                                   |