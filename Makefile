include .env

SUFFIX?=""
export USER_ID      := $(shell id -u)
export USER_NAME    := $(shell whoami)
export PROJECT_DIR  := $(shell pwd)
export COMPOSE_CMD  := docker-compose -f docker/docker-compose.yaml -p ${PROJECT_NAME}${SUFFIX}-${USER_NAME}

ifeq (${GPU_IDS}, none)
	export RUNTIME := runc
else
	export RUNTIME := nvidia
endif

.PHONY: build
build:
	$(COMPOSE_CMD) build

.PHONY: logs
logs:
	${COMPOSE_CMD} logs

.PHONY: up
up:
	$(COMPOSE_CMD) up --detach

.PHONY: down
down:
	$(COMPOSE_CMD) down

.PHONY: connect
connect:
	make up
	${COMPOSE_CMD} exec jupyter bash

.PHONY: shell
shell:
	${COMPOSE_CMD} exec repl bash

