#!/bin/bash
conda activate yolov5
python ./inference-tool.py -i "$1" "$2"
echo "hello!"
