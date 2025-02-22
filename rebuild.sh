#!/bin/bash

rm -rf build &&
    cmake -B build . -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_VERBOSE_MAKEFILE=ON &&
    make -C build -j 24 install &&
    time (cd dev && python3 -m unittest -v)
