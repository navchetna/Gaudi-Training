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

## Usage

The list of all possible arguments can be obtained running:
```bash
python optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py --help
```

## Single-card runs

If you want to generate a sequence of text from a prompt of your choice, you should use the `--prompt` argument.
For example:
```bash
python optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--prompt "Here is my prompt"
```

If you want to provide several prompts as inputs, here is how to do it:
```bash
python optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--batch_size 2 \
--prompt "Hello world" "How are you?"
```

If you want to perform generation on default prompts, do not pass the `--prompt` argument.
```bash
python optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample
```

If you want to change the temperature and top_p values, make sure to include the `--do_sample` argument. Here is a sample command.
```bash
python optimum-habana/examples/text-generation/text-generation-pipeline/run_pipeline.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 100 \
--do_sample \
--temperature 0.5 \
--top_p 0.95 \
--batch_size 2 \
--prompt "Hello world" "How are you?"
```

### Variables

| Variable Name                  | Short Explanation                        | Default Value                     |
|--------------------------------|------------------------------------------|-----------------------------------|
| `--model_name_or_path`         | Path/name of the model                   | `meta-llama/Llama-2-7b-hf`        |
| `--use_hpu_graphs`             | Use HPU graphs for optimization          |                                   |
| `--use_kv_cache`               | Whether to use the key/value cache for decoding. |                           |
| `--max_new_tokens`             | Number of tokens to generate.            | 100                               |
| `--do_sample`                  | Whether to use sampling for generation.  |                                   |
| `--temperature`                | Temperature value for text generation    | 0.5                               |
| `--top_p`                      | Top_p value for generating text via sampling | 0/95                          |
| `--batch_size`                 | Input batch size.                        | 2                                 |
| `--prompt`                     | To give a prompt of your choice as input.|                                   |
---