#! /usr/bin/env bash

mkdir -p ./bins

g++ -I ./baseline/eigen/ ./baseline/eigen.cpp -o ./bins/eigen

./bins/eigen
