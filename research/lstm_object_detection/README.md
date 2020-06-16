![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Tensorflow Mobile Video Object Detection

Tensorflow mobile video object detection implementation proposed in the
following papers:

<p align="center">
  <img src="g3doc/lstm_ssd_intro.png" width=640 height=360>
</p>

```
"Mobile Video Object Detection with Temporally-Aware Feature Maps",
Liu, Mason and Zhu, Menglong, CVPR 2018.
```
\[[link](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Mobile_Video_Object_CVPR_2018_paper.pdf)\]\[[bibtex](
https://scholar.googleusercontent.com/scholar.bib?q=info:hq5rcMUUXysJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAXLdwXcU5g_wiMQ40EvbHQ9kTyvfUxffh&scisf=4&ct=citation&cd=-1&hl=en)\]


<p align="center">
  <img src="g3doc/Interleaved_Intro.png" width=480 height=360>
</p>

```
"Looking Fast and Slow: Memory-Guided Mobile Video Object Detection",
Liu, Mason and Zhu, Menglong and White, Marie and Li, Yinxiao and Kalenichenko, Dmitry
```
\[[link](https://arxiv.org/abs/1903.10172)\]\[[bibtex](
https://scholar.googleusercontent.com/scholar.bib?q=info:rLqvkztmWYgJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAXLdwNf-LJlm2M1ymQHbq2wYA995MHpJu&scisf=4&ct=citation&cd=-1&hl=en)\]


## Maintainers
* masonliuw@gmail.com
* yinxiao@google.com
* menglong@google.com
* yongzhe@google.com


## Table of Contents

  * <a href='g3doc/exporting_models.md'>Exporting a trained model</a>

---

# Usage

## Docker image
tensorflow/tensorflow:1.14.0-gpu-py3

## Setup
Follow this guide written for object detection ignoring the part about tensorflow installation: https://github.com/ultinous-bjenei/models/blob/master/research/object_detection/g3doc/installation.md

When compiling proto files, prefer manual protobuf-compiler installation, and include __lstm_object_detection/protos/*.proto__ along with __object_detection/protos/*.proto__

Generate input data tfrecord file with __create_imagenet_det_tfrecord.py__

The __Looking Fast and Slow__ configuration file __configs/lstm_ssd_interleaved_mobilenet_v2_imagenet.config__ needs to be modified to use a training epoch limit (not specified in the paper), cosine lr decay, existing tfrecord dataset paths and label map path included in this directory as __imagenet_det_label_map.pbtxt__

The input_reader section also requires changes to enable thorough shuffling.

## Train
```bash
base="`pwd`/.."
base="`realpath "$base"`"
export PYTHONPATH="$base:$base/slim"
python train.py \
--train_dir=train_dir \
--pipeline_config_path=configs/lstm_ssd_interleaved_mobilenet_v2_imagenet.config
```

## Eval
```bash
base="`pwd`/.."
base="`realpath "$base"`"
export PYTHONPATH="$base:$base/slim"
python eval.py \
--run_once \
--checkpoint_dir=train_dir \
--eval_dir=eval_dir \
--pipeline_config_path=configs/lstm_ssd_interleaved_mobilenet_v2_imagenet.config
```

## Debug
```bash
tensorboard --logdir=train_dir
tensorboard --logdir=eval_dir
```
tf.summary.text and tf.summary.image can come handy

## Issue link
https://github.com/tensorflow/models/issues/6253
