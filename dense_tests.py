#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import tensorflow as tf
import _dense_grad
dense_module = tf.load_op_library('build/libdense.so')

class DenceOpTest(unittest.TestCase):
    def test_dense(self):
        with tf.Session(''):
            print("Test begin")
            #input feature width 1
            #batch count 2
            #units 3
            d_input = [[1], [2]]
            d_W = [[1, 2, 3]]
            d_b = [[0.1,0.2,0.3]]
            print("Injecting")
            print(d_input)
            print(d_W)
            print(d_b)
            d = dense_module.dense(d_input, d_W, d_b)
            print("dense",d)
            ret = d.eval()
            print("out:", ret)
            print("Test end")

    def test_denseGrad(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float32, shape = (2,1))
            print(x)
            W = tf.constant(np.asarray([[1,2,3]]).astype(np.float32))
            b = tf.constant(np.asarray([[0.1,0.2,0.3]]).astype(np.float32))

            Wx_dense = dense_module.dense(x, W, b)
            grad_x_dense = tf.gradients(Wx_dense, x)
            
            gradient_dense = sess.run(grad_x_dense, feed_dict = {x: np.asarray([[1], [2]]).astype(np.float32)})
            
            print(gradient_dense)

            #self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
            #self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])
    
                
if __name__ == '__main__':
    unittest.main()