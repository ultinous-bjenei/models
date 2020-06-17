import tensorflow as tf
from sys import argv
from google.protobuf.json_format import MessageToJson

assert len(argv) == 2

l = list(tf.python_io.tf_record_iterator(argv[1]))
input(str(len(l)))
for example in l:
    input(MessageToJson(tf.train.SequenceExample.FromString(example)))
