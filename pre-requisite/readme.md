# Intel Gaudi Accelerator Container Setup Guide

## 1. Prerequisites

### Create docker container

```bash
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host"
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all $DOCKER_OPTS vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest my-container
```
Configure the number of cards to use:
```bash
-e HABANA_VISIBLE_DEVICES=0,1,2,3 # Will use 4 cards
```

**Huggingface Access Requirements:**
- For gated HuggingFace repositories:
  - [Video Guide](https://youtu.be/p0qAdKhufoo?si=CHLlwRJltkSIqYsE)
  - [Documentation](https://huggingface.co/docs/hub/models-gated)
  


### Container Access
```bash
docker exec -it my-container /bin/bash
```


### Device Verification
```bash
hl-smi # Should show all available Gaudi devices
```
