#!/bin/bash

make -C build -j 16 install &&
    time (cd dev && python3 -m unittest -v)
