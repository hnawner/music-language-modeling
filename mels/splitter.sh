#!/bin/bash

shuf -zen620 folk/* | xargs -0 mv -t folk_test

