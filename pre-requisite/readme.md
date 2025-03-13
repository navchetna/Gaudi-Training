# Pre Requisite

## Create docker container

```
DOCKER_OPTS="-e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host"
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all $DOCKER_OPTS vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest my-container
```

## Exec into docker container
```
docker exec -it my-container /bin/bash
```

## Check GPU cards status

```
hl-smi
```
