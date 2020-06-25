DOCKER_IMAGE = "kerassimclrv2:latest"
GPU_NUMBERS = $(shell nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
PROJECT_NAME = "keras-simclrv2"


build:
	docker build -t $(DOCKER_IMAGE) dockerfile

train:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm \
	           -v $(shell pwd):/$(PROJECT_NAME)/ $(DOCKER_IMAGE) \
			   python /$(PROJECT_NAME)/$(FILENAME)

ipython:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm \
	           -v $(shell pwd):/$(PROJECT_NAME)/ $(DOCKER_IMAGE) \
			   ipython

jupyterlab:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm -p 8888:8888 \
	           -v $(shell pwd):/$(PROJECT_NAME)/ $(DOCKER_IMAGE) \
			   /run_jupyter.sh --allow-root

bash:
	docker run -it --shm-size=1g --ulimit memlock=-1 --rm \
	           -v $(shell pwd):/$(PROJECT_NAME)/ $(DOCKER_IMAGE) \
			   bash

tensorboard:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm -p 6006:6006 \
	           -v $(shell pwd):/$(PROJECT_NAME)/ $(DOCKER_IMAGE) \
			   tensorboard --logdir /sdi_mrcnn/$(LOGDIR) --bind_all
