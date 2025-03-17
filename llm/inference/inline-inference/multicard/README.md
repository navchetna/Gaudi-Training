# Text-Generation Pipeline

The text-generation pipeline can be used to perform text-generation by providing single or muliple prompts as input.

## Requirements

### Run the Intel Gaudi Docker image:

```bash
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host"
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all $DOCKER_OPTS vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
cd root
```

### Install Optimum Habana

```bash
git clone https://github.com/huggingface/optimum-habana.git
pip install ./optimum-habana
```

### Install DeepSpeed 

```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```

## Usage

The list of all possible arguments can be obtained running:
```bash
python optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py --help
```

## Multi-card runs

To run a large model such as Llama-2-70b via DeepSpeed, run the following command.
```bash
python optimum-habana/examples/gaudi_spawn.py --use_deepspeed --world_size 8 optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--max_new_tokens 100 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--batch_size 4 \
--prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

To change the temperature and top_p values, run the following command.
```bash
python optimum-habana/examples/gaudi_spawn.py --use_deepspeed --world_size 8 optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-70b-hf \
--max_new_tokens 100 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache \
--do_sample \
--temperature 0.5 \
--top_p 0.95 \
--batch_size 4 \
--prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

### Note 
1. Based on the number of cards you would like to utilize `--world_size` parameter can be changed 

### Variables

| Variable Name                  | Short Explanation                        | Default Value                     |
|--------------------------------|------------------------------------------|-----------------------------------|
| `--world_size`                 | Number of cards to utilize               | 8                                 |
| `--use_mpi`                    | Use DeepSpeed for distributed training   |                                   |
| `--model_name_or_path`         | Path/name of the model                   | `meta-llama/Llama-2-70b-hf`       |
| `--use_hpu_graphs`             | Use HPU graphs for optimization          |                                   |
| `--use_kv_cache`               | Whether to use the key/value cache for decoding. |                           |
| `--max_new_tokens`             | Number of tokens to generate.            | 100                               |
| `--do_sample`                  | Whether to use sampling for generation.  |                                   |
| `--temperature`                | Temperature value for text generation    | 0.5                               |
| `--top_p`                      | Top_p value for generating text via sampling | 0.95                          |
| `--batch_size`                 | Input batch size.                        | 2                                 |
| `--prompt`                     | To give a prompt of your choice as input.|                                   |
---