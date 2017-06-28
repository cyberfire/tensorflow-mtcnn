# tensorflow-mtcnn
Inference only for MTCNN face detector on Tensorflow. 

Inference only for MTCNN face detector on Tensorflow.

Based on davidsandberg's facenet project:

Python version is in the root directory

C++ version  is under directory  cpp/tf_embedded. 
Please checkout README.md in that directory for C++ usage.


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


