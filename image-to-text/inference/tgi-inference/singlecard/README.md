# Llava-v1.6-Mistral-7B on 1 card with BF16 Precision

## Pull the official Docker image with:

```bash
docker pull ghcr.io/huggingface/tgi-gaudi:2.3.1
```

## Launch a local server instance

```bash
model=llava-hf/llava-v1.6-mistral-7b-hf
volume=/mnt/hf_cache   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e no_proxy=$no_proxy \
   -e NO_PROXY=$NO_PROXY \
   -e http_proxy=$http_proxy \
   -e https_proxy=$https_proxy \
   -e HTTP_PROXY=$HTTP_PROXY \
   -e HTTPS_PROXY=$HTTPS_PROXY \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e HF_HUB_ENABLE_HF_TRANSFER=1 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
    -e PREFILL_BATCH_BUCKET_SIZE=1 \
    -e BATCH_BUCKET_SIZE=1 \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.3.1 \
   --model-id $model \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-total-tokens 32768 

```

## Wait for the TGI-Gaudi server to come online. You will see something like so:

> 2024-05-22T19:31:48.302239Z  INFO text_generation_router: router/src/main.rs:378: Connected

## You can then send a simple request to the server from a separate terminal:

```bash
curl -N 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)What is this a picture of?\n\n","parameters":{"max_new_tokens":16, "seed": 42}}' \
    -H 'Content-Type: application/json'
```

## Adjusting TGI Parameters

Maximum sequence length is controlled by two arguments:
- `--max-input-tokens` is the maximum possible input prompt length. Default value is `4095`.
- `--max-total-tokens` is the maximum possible total length of the sequence (input and output). Default value is `4096`.

Maximum batch size is controlled by two arguments:
- For prefill operation, please set `--max-batch-prefill-tokens` as `bs * max-input-tokens`, where `bs` is your expected maximum prefill batch size.
- For decode operation, please set `--max-batch-total-tokens` as `bs * max-total-tokens`, where `bs` is your expected maximum decode batch size.
- Please note that batch size will be always padded to the nearest multiplication of `BATCH_BUCKET_SIZE` and `PREFILL_BATCH_BUCKET_SIZE`.

To ensure greatest performance results, at the beginning of each server run, warmup is performed. It's designed to cover major recompilations while using HPU Graphs. It creates queries with all possible input shapes, based on provided parameters (described in this section) and runs basic TGI operations on them (prefill, decode, concatenate).

Except those already mentioned, there are other parameters that need to be properly adjusted to improve performance or memory usage:

- `PAD_SEQUENCE_TO_MULTIPLE_OF` determines sizes of input length buckets. Since warmup creates several graphs for each bucket, it's important to adjust that value proportionally to input sequence length. Otherwise, some out of memory issues can be observed.
- `ENABLE_HPU_GRAPH` enables HPU graphs usage, which is crucial for performance results. Recommended value to keep is `true` .

For more information and documentation about Text Generation Inference, checkout [the README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.

## Environment Variables

| Name                        | Value(s)   | Default          | Description                                                                                                                      | Usage                        |
| --------------------------- | :--------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| ENABLE_HPU_GRAPH            | True/False | True             | Enable hpu graph or not                                                                                                          | add -e in docker run command |
| LIMIT_HPU_GRAPH             | True/False | False            | Skip HPU graph usage for prefill to save memory, set to `True` for large sequence/decoding lengths(e.g. 300/212)                 | add -e in docker run command |
| BATCH_BUCKET_SIZE           | integer    | 8                | Batch size for decode operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs  | add -e in docker run command |
| PREFILL_BATCH_BUCKET_SIZE   | integer    | 4                | Batch size for prefill operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs | add -e in docker run command |
| PAD_SEQUENCE_TO_MULTIPLE_OF | integer    | 128              | For prefill operation, sequences will be padded to a multiple of provided value.                                                 | add -e in docker run command |
| SKIP_TOKENIZER_IN_TGI       | True/False | False            | Skip tokenizer for input/output processing                                                                                       | add -e in docker run command |
| WARMUP_ENABLED              | True/False | True             | Enable warmup during server initialization to recompile all graphs. This can increase TGI setup time.                            | add -e in docker run command |
| QUEUE_THRESHOLD_MS          | integer    | 120              | Controls the threshold beyond which the request are considered overdue and handled with priority. Shorter requests are prioritized otherwise.                            | add -e in docker run command |
| USE_FLASH_ATTENTION         | True/False | False            | Whether to enable Habana Flash Attention, provided that the model supports it. Currently only llama and mistral supports this feature. Please refer to https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html?highlight=fusedsdpa#using-fused-scaled-dot-product-attention-fusedsdpa |
| FLASH_ATTENTION_RECOMPUTE   | True/False | False            | Whether to enable Habana Flash Attention in recompute mode on first token generation. |
