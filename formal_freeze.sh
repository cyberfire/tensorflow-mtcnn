#!/bin/bash

OUTPUT_NODE="--output_node_names pnet/conv4-2/BiasAdd,pnet/prob1,rnet/conv5-2/conv5-2,rnet/prob1,onet/prob1,onet/conv6-2/conv6-2,onet/conv6-3/conv6-3"
python freeze_graph.py --input_graph ./models/save/graph.pb --input_checkpoint ./models/save/mtcnn --output_graph=./models/save/freeze_model.pb $OUTPUT_NODE
