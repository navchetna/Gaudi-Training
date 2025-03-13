# Finetuning CTC Models - Single Card

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

## Connectionist Temporal Classification

The Connectionist Temporal Classification (CTC) model is widely used in Automatic Speech Recognition (ASR) to align input audio sequences with output text without requiring explicit frame-level alignment. It enables end-to-end speech recognition by predicting probabilities over characters or words at each time step while allowing flexible alignment through a blank token and repetition collapsing.

Following command shows how to fine-tune [wav2vec2-large-lv60](https://huggingface.co/facebook/wav2vec2-large-lv60) on [Librispeech](https://huggingface.co/datasets/librispeech_asr) using a single HPU.

```sh
python optimum-habana/examples/speech-recognition/run_speech_recognition_ctc.py \
    --dataset_name="librispeech_asr" \
    --model_name_or_path="facebook/wav2vec2-large-lv60" \
    --dataset_config_name="clean" \
    --train_split_name="train.100" \
    --eval_split_name="validation" \
    --output_dir="/tmp/wav2vec2-librispeech-clean-100h-demo-dist" \
    --preprocessing_num_workers="64" \
    --dataloader_num_workers 8 \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --per_device_train_batch_size="4" \
    --learning_rate="3e-4" \
    --warmup_steps="500" \
    --text_column_name="text" \
    --layerdrop="0.0" \
    --freeze_feature_encoder \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --gaudi_config_name="Habana/wav2vec2" \
    --throughput_warmup_steps="3" \
    --sdp_on_bf16 \
    --bf16 \
    --use_hpu_graphs_for_training \
    --use_hpu_graphs_for_inference \
    --attn_implementation sdpa \
    --trust_remote_code "True"
```

### Variables

| Variable Name                        | Short Explanation                          | Default Value                     |
|--------------------------------------|--------------------------------------------|-----------------------------------|
| `--dataset_name`                     | Name of the dataset                        | `"librispeech_asr"`               |
| `--model_name_or_path`               | Path/name of the model                     | `"facebook/wav2vec2-large-lv60"` |
| `--dataset_config_name`              | Configuration name for the dataset         | `"clean"`                         |
| `--train_split_name`                 | Name of the training split                 | `"train.100"`                     |
| `--eval_split_name`                  | Name of the evaluation split               | `"validation"`                    |
| `--output_dir`                       | Directory to save model outputs            | `"/tmp/wav2vec2-librispeech-clean-100h-demo-dist"` |
| `--preprocessing_num_workers`        | Number of workers for preprocessing        | `"64"`                            |
| `--dataloader_num_workers`           | Number of workers for data loading         | 8                                 |
| `--overwrite_output_dir`             | Overwrite the output directory             |                                   |
| `--num_train_epochs`                 | Number of training epochs                  | `"3"`                             |
| `--per_device_train_batch_size`      | Batch size per device for training         | `"4"`                             |
| `--learning_rate`                    | Learning rate for optimization             | `"3e-4"`                          |
| `--warmup_steps`                     | Number of warmup steps for LR              | `"500"`                           |
| `--text_column_name`                 | Column name for text data                  | `"text"`                          |
| `--layerdrop`                        | Layer dropout probability                  | `"0.0"`                           |
| `--freeze_feature_encoder`           | Freeze the feature encoder                 |                                   |
| `--chars_to_ignore`                  | Characters to ignore in processing         | `, ? . ! - \; \: \" “ % ‘ ”`     |
| `--do_train`                         | Perform training                           |                                   |
| `--do_eval`                          | Perform evaluation                         |                                   |
| `--use_habana`                       | Enable Habana hardware support             |                                   |
| `--use_lazy_mode`                    | Use lazy mode for computation              |                                   |
| `--gaudi_config_name`                | Gaudi configuration name                   | `"Habana/wav2vec2"`               |
| `--throughput_warmup_steps`          | Steps for throughput warmup                | `"3"`                             |
| `--sdp_on_bf16`                      | Use SDP with BF16 precision                |                                   |
| `--bf16`                             | Use BF16 precision for computation         |                                   |
| `--use_hpu_graphs_for_training`      | Use HPU graphs for training optimization   |                                   |
| `--use_hpu_graphs_for_inference`     | Use HPU graphs for inference optimization  |                                   |
| `--attn_implementation`              | Attention implementation type              | `sdpa`                            |
