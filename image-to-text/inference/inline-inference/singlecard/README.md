# Image to Text 

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


## Single card inference with BF16

To run inference

```sh
python3 optimum-habana/examples/image-to-text/run_pipeline.py \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --use_hpu_graphs \
    --bf16
```

### Note
1. The model **meta-llama/Llama-3.2-11B-Vision-Instruct** requires special access permissions. Please request access on the model's page on Hugging Face.

### Variables 

| Variable Name                  | Short Explanation                        | Default Value                     |
|--------------------------------|------------------------------------------|-----------------------------------|
| `--model_name_or_path`         | Path/name of the model                   | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| `--use_hpu_graphs`             | Use HPU graphs for optimization          |                                   |
| `--bf16`                       | Use BF16 precision for computation       |                                   |