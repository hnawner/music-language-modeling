#!/bin/bash

mkdir folk_minor/

FILES=folk/*

for file in $FILES
do
    if grep -q "Minor" "${file}"
    then
        mv -v "${file}" "folk_minor/"
    fi
done
