#!/bin/bash
#
# Run training (and evaluation apparently)
#
cd models/research

protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/slim"
python3 object_detection/model_main.py \
        --model_dir="../../object_detection_models" \
        --pipeline_config_path="../../ssdlite_mobilenet_v2_coco.config"
