# Pretraining of Llama2-7B using FP8 on the Intel® Gaudi® 2 AI Accelerator

This example will show will show how to run pretraining of Meta Llama2 7B, using the Megatron-LM library, on the Intel Gaudi Accelerator. Megatron-LM serves as a research-oriented framework leveraging Megatron-Core for large language model (LLM) training.

You will learn how to setup the environment, select parameters, execute the workload and then see a price-performance comparison. Intel Gaudi supports PyTorch as the main framework for Training.

This tutorial will be based on the Habana implementation of [Megatron-LM repository](https://github.com/HabanaAI/Megatron-LM).

The following steps will let you run pretraining on the Llama 7B. In the next sections each step will be described step-by-step:

- Run the Intel Gaudi PyTorch Docker image; this ensures that all the SW is installed and configured properly.
- Install pre-requisites.
- Download and pre-process dataset.
- Select parameters and run pretraining on the model.

**This tutorial assumes you have access to an Intel Gaudi node**

## Docker Setup

Use the latest Intel Gaudi docker image by first calling the docker run command which will automatically download and run the docker:
```bash
docker run -itd --name Gaudi_Docker --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.19.0/ubuntu22.04/habanalabs/pytorch-installer-2.5.1:latest
```

We then start the docker and enter the docker environment by issuing the following command:
```bash
docker exec -it Gaudi_Docker bash
```

## Model Setup

Now that we’re running in a docker environment, we can now install the remaining libraries and model repositories: Start in the root directory and install the Megatron-LM Library.
```bash
cd /root
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```

Now install the Hugging Face Optimum Habana library and clone the Megatron-LM repository, notice that we’re selecting the latest validated release of Optimum Habana:
```bash
pip install optimum-habana==1.15.0
git clone -b 1.19.0 https://github.com/HabanaAI/Megatron-LM.git
```

Next, we transition to the Megatron-LM directory and install the set of requirements to perform training:
```bash
cd /root/Megatron-LM
pip install -r megatron/core/requirements.txt
```

Setup the correct path for Megatron-LM:
```bash
echo 'export MEGATRON_LM_ROOT=/root/Megatron-LM' >> ~/.bashrc
source ~/.bashrc
```

Finally, Set Python 3.10 as the default Python version:
```bash
echo 'export PYTHON=/usr/bin/python3.10' >> ~/.bashrc
source ~/.bashrc
```

## How to download the dataset

To download datasets used for training Llama2, you can follow directions in the Megatron-Deepspeed Github page, which show steps to download and preprocess the Oscar-en dataset. This dataset is big, and it will take considerable time to download and preprocess. For this tutorial, we will use a smaller dataset, the customized RedPajama dataset, which will download and prepare much faster, with the purpose to illustrate the pre-training flow.

First, download the redpajama dataset list file, then pick only the first jsonl file, which is arxiv:
```bash
cd /root
mkdir -p redpajama
cd redpajama
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
head -n 1 urls.txt > first_jsonl.txt
```

Next, download the arxiv subset:
```bash
mkdir arxiv
wget -P arxiv/ https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_023827cd-7ee8-42e6-aa7b-661731f4c70f.jsonl
```

We also need to download the tokenizer file correspondent to the Llama7B model:
```bash
wget -O tokenizer.model "https://huggingface.co/huggyllama/llama-7b/resolve/main/tokenizer.model"
```

The last step is to install the modules needed for data preparation and complete the pre-processing step:
```bash
cd /root/redpajama/
pip install nltk sentencepiece
mkdir -p arxiv_tokenized
wget -P arxiv_tokenized https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget -P arxiv_tokenized https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
python $MEGATRON_LM_ROOT/tools/preprocess_data.py --input arxiv/*.jsonl \
      --output-prefix arxiv_tokenized/meg-gpt2 --tokenizer-model ./tokenizer.model \
      --append-eod --tokenizer-type GPT2BPETokenizer --workers 64 --vocab-file arxiv_tokenized/gpt2-vocab.json --merge-file arxiv_tokenized/gpt2-merges.txt
```

## Running Llama2 7B Pretraing Using the FP8 Datatype

We are now ready to start running pretraining on this model.
```bash
export MEGATRON_LM_ROOT='/root/Megatron-LM'
echo $MEGATRON_LM_ROOT
export LOG_LEVEL_ALL=4
export ENABLE_CONSOLE=true
export HABANA_LOGS=./habana_log
export MEGATRON_LM_ROOT=/root/Megatron-LM/
export MODEL_REFERENCES_ROOT=/root/Megatron-LM/
export HL_DATA_DIR_ROOT=/root/redpajama/arxiv_tokenized/
export HL_DATA_FILE_PREFIX=meg-gpt2_text_document
export OUT_DIR=Llama2-7B-training
export HL_HOSTSFILE=/launch/hostsfile
mkdir -p ${OUT_DIR}
HL_SAVE=0 \
HL_EXIT_INTERVAL=80 \
HL_RESULTS_DIR=${OUT_DIR} \
HL_LOG_INTERVAL=10 \
HL_TOKENIZER_TYPE=GPT2BPETokenizer \
HL_DATA_DIR_ROOT=${HL_DATA_DIR_ROOT} \
HL_DATA_FILE_PREFIX=$HL_DATA_FILE_PREFIX \
HL_GBS=1024 \
HL_LLAMA_VER=2 \
HL_LLAMA_MODEL_SIZE=7 \
HL_NUM_NODES=1 \
HL_PP=1 HL_TP=1 HL_DP=8 \
HL_CKP_ACT=2 \
HL_SEQ_LEN=4096 \
HL_ZERO_STAGE=1 \
HL_USE_FAST_SOFTMAX=1 \
HL_GRAD_ACCUM_DTYPE=bf16  \
HL_USE_TRANSFORMER_ENGINE=1 \
HL_USE_CACHE_FP8_WEIGHT_FWD=1 \
HL_USE_CACHE_FP8_WEIGHT=1 \
${MODEL_REFERENCES_ROOT}/examples/llama/pretrain_llama.sh 2>&1 | tee ${OUT_DIR}/llama_8x.log
```

The performance results can vary depending on hardware used.

## Next Steps

Now that you have run a pretraining case, you can go back to the Hugging Face Optimum Habana [validated models](https://github.com/huggingface/optimum-habana?tab=readme-ov-file#validated-models) to see more options for running training or inference.
