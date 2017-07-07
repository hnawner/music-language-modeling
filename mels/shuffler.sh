#!/bin/bash

maj="$(ls "pitches/folk_major" | wc -l)"
min="$(ls "pitches/folk_minor" | wc -l)"
div=10

maj=$((maj / div))
min=$((min / div))

shuf -ze --head-count=${maj} pitches/folk_major/* | xargs -0 mv -t pitches/folk_maj_test
shuf -ze --head-count=${min} pitches/folk_minor/* | xargs -0 mv -t pitches/folk_min_test
