import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
dense_grad_module = tf.load_op_library('build/libdense.so')

@ops.RegisterGradient("Dense")
def _dense_grad_cc(op, grad):
    return dense_grad_module.dense_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])