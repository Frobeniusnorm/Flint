curl https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet50-v1-12.tar.gz?download=true --output resnet50-v1-12.tar.gz
tar -xf resnet50-v1-12.tar.gz
rm resnet50-v1-12.tar.gz
mv resnet50-v1-12/* ./
rm -r resnet50-v1-12
