#!/bin/bash

FILES=rhythms/train/*

for file in $FILES
do
    if grep -q -e "simple duple" -e "simple quadruple" ${file}
    then
        mv -v ${file} rhythms/folk_duple/
    else
        mv -v ${file} rhythms/folk_other/
    fi
done
