# Image to Text 

This directory contains a script that showcases how to perform image to text generation on Intel® Gaudi® AI Accelerators using a single card.

## Requirements 

Install Optimum Habana

```sh
git clone https://github.com/huggingface/optimum-habana.git

pip install ./optimum-habana
```
Install additional task specific requirements

```sh
pip install -r optimum-habana/examples/image-to-text/requirements.txt
```


## Single card inference with BF16

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