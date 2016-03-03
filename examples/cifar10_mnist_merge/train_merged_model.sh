#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10_mnist_merge/merge_model_solver.prototxt
