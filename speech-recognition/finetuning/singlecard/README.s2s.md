# Finetuning Seq2Seq Models - Single Card

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
pip install -r optimum-habana/examples/speech-recognition/requirements.txt
```


## Sequence to Sequence

A **Sequence-to-Sequence (Seq2Seq) model** in **Automatic Speech Recognition (ASR)** maps an input audio sequence to a text sequence using an **encoder-decoder architecture**, often with attention mechanisms. Unlike CTC, Seq2Seq models generate outputs autoregressively, making them more powerful for capturing context but also more computationally intensive.

The following example shows how to fine-tune the [Whisper small](https://huggingface.co/openai/whisper-small) checkpoint on the Hindi subset of [Common Voice 11](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) using a single HPU device in bf16 precision:

```sh
python optimum-habana/examples/speech-recognition/run_speech_recognition_seq2seq.py \
    --model_name_or_path="openai/whisper-small" \
    --dataset_name="mozilla-foundation/common_voice_11_0" \
    --dataset_config_name="hi" \
    --language="hindi" \
    --task="transcribe" \
    --train_split_name="train+validation" \
    --eval_split_name="test" \
    --gaudi_config_name="Habana/whisper" \
    --max_steps="625" \
    --output_dir="/tmp/whisper-small-hi" \
    --per_device_train_batch_size="48" \
    --per_device_eval_batch_size="2" \
    --logging_steps="25" \
    --learning_rate="1e-5" \
    --warmup_steps="100" \
    --eval_strategy="steps" \
    --eval_steps="400" \
    --save_strategy="steps" \
    --save_steps="200" \
    --generation_max_length="225" \
    --preprocessing_num_workers="1" \
    --max_duration_in_seconds="30" \
    --text_column_name="sentence" \
    --freeze_feature_encoder="False" \
    --sdp_on_bf16 \
    --bf16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --use_habana \
    --use_hpu_graphs_for_inference \
    --label_features_max_length 128 \
    --dataloader_num_workers 8 \
    --throughput_warmup_steps 3 \
    --sdp_on_bf16 \
    --trust_remote_code "True"
```

### Variables 

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--model_name_or_path`               | Path/name of the model                     | `"openai/whisper-small"`          |
| `--dataset_name`                     | Name of the dataset                        | `"mozilla-foundation/common_voice_11_0"` |
| `--dataset_config_name`              | Configuration name for the dataset         | `"hi"`                            |
| `--language`                         | Language for processing                    | `"hindi"`                         |
| `--task`                             | Task to perform (e.g., transcribe)         | `"transcribe"`                    |
| `--train_split_name`                 | Name of the training split                 | `"train+validation"`              |
| `--eval_split_name`                  | Name of the evaluation split               | `"test"`                          |
| `--gaudi_config_name`                | Gaudi configuration name                   | `"Habana/whisper"`                |
| `--max_steps`                        | Maximum number of training steps           | `"5000"`                          |
| `--output_dir`                       | Directory to save model outputs            | `"/tmp/whisper-small-hi"`         |
| `--per_device_train_batch_size`      | Batch size per device for training         | `"48"`                            |
| `--per_device_eval_batch_size`       | Batch size per device for evaluation       | `"2"`                             |
| `--logging_steps`                    | Steps between logging updates              | `"25"`                            |
| `--learning_rate`                    | Learning rate for optimization             | `"1e-5"`                          |
| `--warmup_steps`                     | Number of warmup steps for LR              | `"500"`                           |
| `--eval_strategy`                    | Evaluation strategy                        | `"steps"`                         |
| `--eval_steps`                       | Steps between evaluations                  | `"1000"`                          |
| `--save_strategy`                    | Model saving strategy                      | `"steps"`                         |
| `--save_steps`                       | Steps between saves                        | `"1000"`                          |
| `--generation_max_length`            | Max length for generated sequences         | `"225"`                           |
| `--preprocessing_num_workers`        | Number of workers for preprocessing        | `"1"`                             |
| `--max_duration_in_seconds`          | Maximum audio duration in seconds          | `"30"`                            |
| `--text_column_name`                 | Column name for text data                  | `"sentence"`                      |
| `--freeze_feature_encoder`           | Freeze the feature encoder                 | `"False"`                         |
| `--sdp_on_bf16`                      | Use SDP with BF16 precision                |                                   |
| `--bf16`                             | Use BF16 precision for computation         |                                   |
| `--overwrite_output_dir`             | Overwrite the output directory             |                                   |
| `--do_train`                         | Perform training                           |                                   |
| `--do_eval`                          | Perform evaluation                         |                                   |
| `--predict_with_generate`            | Generate predictions during evaluation     |                                   |
| `--use_habana`                       | Enable Habana hardware support             |                                   |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--label_features_max_length`        | Max length for label features              | 128                               |
| `--dataloader_num_workers`           | Number of workers for data loading         | 8                                 |
| `--throughput_warmup_steps`          | Steps for throughput warmup                | 3                                 |