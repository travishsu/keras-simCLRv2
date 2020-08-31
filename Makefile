DOCKER_IMAGE = "kerassimclrv2:latest"
GPU_NUMBERS = $(shell nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

init:
	mkdir logs data

build:
	docker build --no-cache -t $(DOCKER_IMAGE) dockerfile

run:
	docker run -it --rm \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) python /workspace/$(FILENAME)

run-gpus:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) python /workspace/$(FILENAME)

ipython:
	docker run -it --rm \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) ipython

ipython-gpus:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) ipython

jupyterlab:
	docker run -it --rm -p 8888:8888 \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) /run_jupyter.sh --allow-root

jupyterlab-gpus:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm -p 8888:8888 \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) /run_jupyter.sh --allow-root

bash:
	docker run -it --rm \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) bash

bash-gpus:
	docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) bash

tensorboard:
	docker run -it -p 6006:6006 \
			   -v $(shell pwd):/workspace/ \
			   -v $(MNT_DIR):/mnt/ \
			   $(DOCKER_IMAGE) tensorboard --logdir /workspace/$(LOGDIR) --bind_all
