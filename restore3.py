# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# Borrowed from davidsandberg's facenet project: https://github.com/davidsandberg/facenet
# From this directory:
#   facenet/src/align
#
# Just keep the MTCNN related stuff and removed other codes
# python package required:
#     tensorflow, opencv,numpy


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import cv2
import model_load as ld


def restore_mtcnn(sess):

    pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
    rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
    onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

    return pnet_fun, rnet_fun, onet_fun
 

def main(args):
    

    graph=ld.load_graph(args.frozen_model_filename)
    
    sess = tf.Session(graph=graph)

    for op in graph.get_operations():
        print(op.name)

    pnet, rnet, onet = restore_mtcnn(sess)


    minsize = 40 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    
    filename =args.input 
    output_filename =args.output


    draw = cv2.imread(filename)

    img=cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)
    
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]


    for b in bounding_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0))
        print(b)



    for p in points.T:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

    cv2.imwrite(output_filename,draw)
                            
    print('Total %d face(s) detected, saved in %s' % (nrof_faces,output_filename))


            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image to be detected for faces.',default='./test.jpg')
    parser.add_argument('--output', type=str, help='new image with boxed faces',default='new.jpg')
    parser.add_argument("--frozen_model_filename", default="./models/save/frozen_model.pb", type=str, help="Frozen model file to import")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
