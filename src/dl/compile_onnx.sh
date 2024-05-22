#!/usr/bin/bash
# execute when the onnx file needs to be rebuild
protoc onnx.proto3 --cpp_out=.
mv onnx.proto3.pb.cc onnx.proto3.pb.cpp
