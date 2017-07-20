#!/bin/bash

mkdir sample
mels="$(ls "train" | wc -l)"
div=2

mels=$((mels / div))

shuf -ze --head-count=${mels} train/* | xargs -0 cp -t sample
