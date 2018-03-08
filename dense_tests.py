#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.

.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import tensorflow as tf
import _dense_grad
import tensorflow.contrib.slim as slim
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
    
    def test_gradientCheck(self):
        for i in range(10):
            with tf.Session('') as sess:

                units = 1
                samples = 1
                inputs = 1 


                x = tf.constant(np.random.rand(samples,inputs).astype(np.float32)-0.5)
                samples,inputs = x.get_shape()
                W = tf.constant(np.random.rand(inputs,units).astype(np.float32)-0.5)
                b = tf.constant(np.random.rand(1,units).astype(np.float32)-0.5)
                y0 = tf.nn.relu(tf.matmul(x,W)+b)
                y1 = tf.nn.relu(dense_module.dense(x, W, b))
                
               
                
                print(x)
                print(W)
                print(b)
                print(y0)
                

                d = 10e-7



                print("homebrew")
                e= tf.test.compute_gradient_error(x, x.get_shape().as_list(), y0, y0.get_shape().as_list())
                print("x gradient error = %f"%(e))
                e= tf.test.compute_gradient_error(W, W.get_shape().as_list(), y0, y0.get_shape().as_list())
                print("W gradient error = %f"%(e))
                e= tf.test.compute_gradient_error(b, b.get_shape().as_list(), y0, y0.get_shape().as_list())
                print("b gradient error = %f"%(e))
                print("c++")
                e= tf.test.compute_gradient_error(x, x.get_shape().as_list(), y1, y1.get_shape().as_list())
                print("x gradient error = %f"%(e))
                e= tf.test.compute_gradient_error(W, W.get_shape().as_list(), y1, y1.get_shape().as_list())
                print("W gradient error = %f"%(e))
                e= tf.test.compute_gradient_error(b, b.get_shape().as_list(), y1, y1.get_shape().as_list())
                print("b gradient error = %f"%(e))



            #self.assertEqual(gradient_tf[0][0], gradient_inner_product[0][0])
            #self.assertEqual(gradient_tf[0][1], gradient_inner_product[0][1])
    
                
if __name__ == '__main__':
    unittest.main()