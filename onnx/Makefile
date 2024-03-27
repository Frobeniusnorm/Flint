# Just for testing, will change to CMakeFile
onnx_test: onnx_test.o onnx.o
	g++ -o onnx_test onnx_test.o onnx.o `pkg-config --cflags --libs protobuf`

onnx_test.o: test.cpp
	g++ test.cpp -c -o onnx_test.o `pkg-config --cflags --libs protobuf`

onnx.o: onnx.proto3.pb.cc
	g++ onnx.proto3.pb.cc -c -o onnx.o `pkg-config --cflags --libs protobuf`
