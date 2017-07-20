#!/bin/bash

files="melfiles/*"
total=0
count=0

for f in $files; do
    echo $f
    mel_logp=$(./melprob $f)
    mel_logp=$(echo "scale=4; $mel_logp * -1" | bc -lq)
    n_notes=$(grep -c "Note" $f)
    total=$(echo "scale=4; $total + $mel_logp" | bc -lq)
    (( count += n_notes ))
done
avg=$(echo "scale=3; $total / $count" | bc -lq)
echo $avg
