# tensorflow-mtcnn
Inference only for MTCNN face detector on Tensorflow. 

Inference only for MTCNN face detector on Tensorflow.

Based on davidsandberg's facenet project:

Python version is in the root directory

There are two version for C++.

One is to be build inside tensorflow code repository, so that it needs to be copied to the directory tensorflow/example.
please check cpp/tf_embedded/README.md for details.

The other is the standalone one, just needs libtensorflow.so and c_api.h to build and run.
Please check cpp/standalone/README.md for more details

## Python Run
1. install tensorflow first, please refers to https://www.tensorflow.org/install
2. install python packages: opencv, numpy
3. python ./facedetect_mtcnn.py --input input.jpg --output  new.jpg

## Credit

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Tensorflow 

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet
From this directory:
  facenet/src/align


