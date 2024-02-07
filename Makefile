docker_build := .DOCKER
docker_image_name := anterondocker/tensorrt
.PHONY: run

$(docker_build):
	docker build . -t $(docker_image_name)
	touch $(docker_build)

run: $(docker_build)
	docker run --rm -v $(abspath .):/app -it $(docker_image_name)
