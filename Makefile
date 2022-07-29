.PHONY: install
install:
	@bash bin/install.sh

.PHONY: build_base_image 
build_base_image:
	@echo "building base image..."
	@chmod +x bin/build_base_image.sh
	@bash ./bin/build_base_image.sh

.PHONY: build_pipeline
build_pipeline:
	@echo "building vertex pipeline..."
	@chmod +x bin/build_vertex_pipeline.sh
	@bash ./bin/build_vertex_pipeline.sh latest v$(date +%s)
	@echo "vertex pipeline has been built."

build_all: build_base_image build_pipeline

.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: compile
compile:
	@export PYTHONPATH=pipelines \
	&& python3 pipelines/src/run_vertex_pipeline.py \
	--action compile --gcs

.PHONY: run_pipeline
run_pipeline:
	@export PYTHONPATH=pipelines \
	&& python3 pipelines/src/run_vertex_pipeline.py --async --disable-caching \
	--action run_latest --gcs
