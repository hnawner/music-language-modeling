#!/bin/bash

mkdir ../maj_min_split/minor_test/

FILES=../maj_min_split/test/*

for file in $FILES
do
    if grep -q "Minor" "${file}"
    then
        mv -v "${file}" "../maj_min_split/minor_test/"
    fi
done
