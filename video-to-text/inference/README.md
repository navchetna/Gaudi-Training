# Video to Text

This directory contains example scripts that demonstrate how to perform video comprehension on Gaudi with graph mode.

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
pip install -r optimum-habana/examples/video-comprehension/requirements.txt
```

## Single-HPU inference

### Video-LLaVA Model

```sh
python3 optimum-habana/examples/video-comprehension/run_example.py \
    --model_name_or_path "LanguageBind/Video-LLaVA-7B-hf" \
    --warmup 3 \
    --n_iterations 5 \
    --batch_size 1 \
    --use_hpu_graphs \
    --bf16 \
    --output_dir ./
```

### Models that have been validated:
[LanguageBind/Video-LLaVA-7B-hf](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)

### Variables

| Variable Name                  | Short Explanation                        | Default Value                     |
|--------------------------------|------------------------------------------|-----------------------------------|
| `--model_name_or_path`         | Path/name of the model                   | `"LanguageBind/Video-LLaVA-7B-hf"`|
| `--warmup`                     | Number of warmup iterations              | 3                                 |
| `--n_iterations`               | Number of iterations to run              | 5                                 |
| `--batch_size`                 | Batch size for processing                | 1                                 |
| `--use_hpu_graphs`             | Use HPU graphs for optimization          |                                   |
| `--bf16`                       | Use BF16 precision for computation       |                                   |
| `--output_dir`                 | Directory to save outputs                | `./`                              |
| `--video_path`                 | Path to video input                      |                                   |
| `--prompt`                     | Prompt to be given                       |                                   |
