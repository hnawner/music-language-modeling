#!/bin/bash

maj="$(ls "folk_major" | wc -l)"
min="$(ls "folk_minor" | wc -l)"
div=10

maj=$((maj / div))
min=$((min / div))

shuf -ze --head-count=${maj} folk_major/* | xargs -0 mv -t folk_maj_test
shuf -ze --head-count=${min} folk_minor/* | xargs -0 mv -t folk_min_test
