#!/bin/bash

FILES=melfiles/*

for file in $FILES
do
    if grep -q -e "simple duple" -e "simple quadruple" ${file}
    then
        mv -v ${file} rhythms/duple/
    fi
done
