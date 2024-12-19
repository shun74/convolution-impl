#! /bin/bash

FILE_DIR=$(dirname $(realpath $0))
nvcc -shared -Xcompiler -fPIC -o ${FILE_DIR}/conv2d.so ${FILE_DIR}/conv2d.cu
