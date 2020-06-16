import os
import glob
import hashlib
import logging
import argparse
import tensorflow as tf
import numpy as np
from collections import defaultdict
from lxml import etree
from pprint import pprint
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

"""
bjenei: This script is a heavily modified version of:
        https://github.com/wakanda-ai/tf-detectors/blob/master/datasets/VID2015/vid_2015_to_tfrecord.py
        It creates a tfrecord data file containing SequenceExample objects.
"""

stringIndex = dict(
    n01503061=1,
    n01662784=2,
    n01674464=3,
    n01726692=4,
    n02062744=5,
    n02084071=6,
    n02118333=7,
    n02121808=8,
    n02129165=9,
    n02129604=10,
    n02131653=11,
    n02324045=12,
    n02342885=13,
    n02355227=14,
    n02374451=15,
    n02391049=16,
    n02402425=17,
    n02411705=18,
    n02419796=19,
    n02484322=20,
    n02503517=21,
    n02509815=22,
    n02510455=23,
    n02691156=24,
    n02834778=25,
    n02924116=26,
    n02958343=27,
    n03790512=28,
    n04468005=29,
    n04530566=30)

counter = defaultdict(lambda: 0)

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True,
                    help="Root directory to raw VID 2015 dataset.")
parser.add_argument("--output_path", required=True,
                    help="Path to output TFRecord")
parser.add_argument("--set", default="train",
                    help="Convert training set, validation set.")
parser.add_argument("--num_frames", default=4, type=int,
                    help="The number of frame to use")
parser.add_argument("--num_examples", default=-1, type=int,
                    help="The number of video to convert to TFRecord file")
args = parser.parse_args()

SETS = ["train", "val", "test"]


def sample_frames(xml_files):
    length = len(xml_files)
    if length < 4:
        return ()
    keep = (length // args.num_frames) * args.num_frames
    return (np.array(xml_files[:keep])
            .reshape((-1, args.num_frames))
            .tolist())


def gen_record(examples_list, annotations_dir, out_filename, root_dir, _set):
    writer = tf.python_io.TFRecordWriter(out_filename)
    length = len(examples_list)
    c = 0
    for i, example in enumerate(examples_list):
        print(length, i+1)
        # sample frames
        xml_pattern = os.path.join(annotations_dir, example + "/*.xml")
        xml_files = sorted(glob.glob(xml_pattern))
        samples = sample_frames(xml_files)
        for sample in samples:
            c += 1
            dicts = []
            # process per single xml
            for xml_file in sample:
                with tf.gfile.GFile(xml_file, "r") as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                dic = recursive_parse_xml_to_dict(
                    xml)["annotation"]
                dicts.append(dic)
            tf_example = dicts_to_tf_example(dicts, root_dir, _set)
            writer.write(tf_example.SerializeToString())
    writer.close()
    print(c, "videos")


def dicts_to_tf_example(dicts, root_dir, _set):
    """Convert XML derived dict to tf.Example proto."""

    global counter

    # nonsequential data
    folder = dicts[0]["folder"]
    filenames = [dic["filename"] for dic in dicts]
    sha = hashlib.sha256(folder.encode("utf8")).hexdigest()
    height = int(dicts[0]["size"]["height"])
    width = int(dicts[0]["size"]["width"])

    # collected sequential data
    path_s = []
    xmin_s, xmax_s = [], []
    ymin_s, ymax_s = [], []
    string_s, index_s = [], []

    # get image paths
    imgs_dir = os.path.join(root_dir,
                            "Data/VID/{}".format(_set),
                            folder)
    imgs_path = [os.path.join(imgs_dir, filename) + ".JPEG"
                 for filename in filenames]

    # iterate frames
    for data, img_path in zip(dicts, imgs_path):
        # open single frame
        path = img_path.encode("utf8")

        # iterate objects
        xmin, xmax = [], []
        ymin, ymax = [], []
        string, index = [], []
        if "object" in data:
            for obj in data["object"]:
                xmin.append(float(obj["bndbox"]["xmin"]) / width)
                xmax.append(float(obj["bndbox"]["xmax"]) / width)
                ymin.append(float(obj["bndbox"]["ymin"]) / height)
                ymax.append(float(obj["bndbox"]["ymax"]) / height)
                name = obj["name"]
                string.append(name.encode("utf8"))
                index.append(stringIndex[name])
                counter[name] += 1

        path_s.append(feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(
                value=[path])))

        xmin_s.append(feature_pb2.Feature(
            float_list=feature_pb2.FloatList(
                value=xmin)))

        xmax_s.append(feature_pb2.Feature(
            float_list=feature_pb2.FloatList(
                value=xmax)))

        ymin_s.append(feature_pb2.Feature(
            float_list=feature_pb2.FloatList(
                value=ymin)))

        ymax_s.append(feature_pb2.Feature(
            float_list=feature_pb2.FloatList(
                value=ymax)))

        string_s.append(feature_pb2.Feature(
            bytes_list=feature_pb2.BytesList(
                value=string)))

        index_s.append(feature_pb2.Feature(
            int64_list=feature_pb2.Int64List(
                value=index)))

    # nonsequential data
    context = feature_pb2.Features(
        feature={
            "image/format": feature_pb2.Feature(
                bytes_list=feature_pb2.BytesList(
                    value=["jpeg".encode("utf-8")])),

            "image/filename": feature_pb2.Feature(
                bytes_list=feature_pb2.BytesList(
                    value=[folder.encode("utf-8")])),

            "image/key/sha256": feature_pb2.Feature(
                bytes_list=feature_pb2.BytesList(
                    value=[sha.encode("utf-8")])),

            "image/source_id": feature_pb2.Feature(
                bytes_list=feature_pb2.BytesList(
                    value=["ILSVRC_2015".encode("utf-8")])),

            "image/height": feature_pb2.Feature(
                int64_list=feature_pb2.Int64List(
                    value=[height])),

            "image/width": feature_pb2.Feature(
                int64_list=feature_pb2.Int64List(
                    value=[width]))
        })

    # collected sequential data
    feature_lists = feature_pb2.FeatureLists(
        feature_list={
            "image/path": feature_pb2.FeatureList(feature=path_s),
            "bbox/xmin": feature_pb2.FeatureList(feature=xmin_s),
            "bbox/xmax": feature_pb2.FeatureList(feature=xmax_s),
            "bbox/ymin": feature_pb2.FeatureList(feature=ymin_s),
            "bbox/ymax": feature_pb2.FeatureList(feature=ymax_s),
            "bbox/label/index": feature_pb2.FeatureList(feature=index_s),
            "bbox/label/string": feature_pb2.FeatureList(feature=string_s)
        })

    return example_pb2.SequenceExample(context=context,
                                       feature_lists=feature_lists)


def main():
    root_dir = args.root_dir

    if args.set not in SETS:
        raise ValueError("set must be in : {}".format(SETS))

    # Read Example list files
    logging.info("Reading from VID 2015 dataset. ({})".format(root_dir))
    list_file_pattern = "ImageSets/VID/{}*.txt".format(args.set)

    examples_paths = glob.glob(os.path.join(root_dir, list_file_pattern))

    examples_list = []
    for examples_path in examples_paths:
        extension = read_examples_list(examples_path)
        if args.set in ("val", "test"):
            extension = list(set(
                os.sep.join(e.split(os.sep)[:-1]) for e in extension))
        examples_list.extend(extension)
    if args.num_examples > 0:
        examples_list = examples_list[:args.num_examples]

    os.makedirs(args.output_path, exist_ok=True)
    out_filename = os.path.join(args.output_path,
                                "VID_2015-"+args.set+".tfrecord")
    annotations_dir = os.path.join(root_dir,
                                   "Annotations/VID/{}".format(args.set))
    gen_record(examples_list, annotations_dir, out_filename, root_dir,
               args.set)

    pprint(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:50])


def read_examples_list(path):
    """Read list of training or validation examples.

    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.

    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

    Args:
    path: absolute path to examples list file.

    Returns:
    list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(" ")[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
    Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != "object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


if __name__ == "__main__":
    main()
