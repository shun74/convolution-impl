#! /bin/bash

FILE_DIR=$(dirname $(realpath $0))
gcc -shared -fPIC -o ${FILE_DIR}/conv2d.so ${FILE_DIR}/conv2d.c
