import tensorflow as tf
from google.protobuf import text_format

with open('pose_landmark_gpu.pbtxt') as f:
  txt = f.read()
gdef = text_format.Parse(txt, tf.GraphDef())

tf.train.write_graph(gdef, '/tmp', 'myfile.pb', as_text=False)