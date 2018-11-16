#!/bin/bash
cat /scratch/darsh/dialogue_systems/6864_project/baseline/tao_model.txt /scratch/darsh/dialogue_systems/6864_project/baseline/tao_model_toy_addendum.txt  > /scratch/darsh/dialogue_systems/6864_project/baseline/tao_model_toy.txt
cp /scratch/darsh/dialogue_systems/6864_project/baseline/test_toy.txt /scratch/darsh/dialogue_systems/tao_model/test_toy.txt
cp /scratch/darsh/dialogue_systems/6864_project/baseline/tao_model_toy.txt /scratch/darsh/dialogue_systems/tao_model/tao_model_toy.txt
python /scratch/darsh/dialogue_systems/tao_model/rcnn/code/pt/main.py --corpus /scratch/darsh/dialogue_systems/tao_model/tao_model_toy.txt --embeddings /scratch/darsh/dialogue_systems/tao_model/apple_corpus_vectors.txt --train /scratch/darsh/dialogue_systems/tao_model/test_toy.txt --model /scratch/darsh/dialogue_systems/tao_model/rcnn/code/pt/model_1 -d 400 --dropout 0.1
