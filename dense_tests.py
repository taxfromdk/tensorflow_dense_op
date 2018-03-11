import os
import unittest
import numpy as np
import tensorflow as tf
import _dense_grad
import tensorflow.contrib.slim as slim
dense_module = tf.load_op_library('build/libdense.so')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class DenceOpTest(unittest.TestCase):
    
    def test_dense(self):
        #input feature width 1
        #batch count 2
        #units 3
        
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float64, shape = (2,1))
            W = tf.constant(np.asarray([[1,2,3]]).astype(np.float64))
            b = tf.constant(np.asarray([[0.1,0.2,0.3]]).astype(np.float64))
            dense = dense_module.dense(x, W, b)
            o = sess.run(dense, feed_dict = {x: np.asarray([[1], [2]]).astype(np.float64)})
            print(o)
            

            d_input = [[1], [2]]
            d_W = [[1, 2, 3]]
            d_b = [[0.1,0.2,0.3]]
            d = dense_module.dense(d_input, d_W, d_b)
            o = d.eval()
            print(o)

    def test_denseGrad_cpu(self):
        with tf.Session('') as sess:
            x = tf.placeholder(tf.float64, shape = (2,1))
            W = tf.constant(np.asarray([[1,2,3]]).astype(np.float64))
            b = tf.constant(np.asarray([[0.1,0.2,0.3]]).astype(np.float64))
            Wx_dense = dense_module.dense(x, W, b)
            grad_x_dense = tf.gradients(Wx_dense, x)
            gradient_dense = sess.run(grad_x_dense, feed_dict = {x: np.asarray([[1], [2]]).astype(np.float64)})
    

    
    def test_gradientCheck_cpu(self):
        with tf.Session('') as sess:
            for device in ['/cpu:0', '/gpu:0']:
                with tf.device(device):
                    #decide dimensions
                    units = 2
                    samples = 3
                    inputs = 4 

                    print("Units:",units,"Samples:",samples,"Inputs:",inputs)
                    
                    n = samples*inputs + inputs*units+units
                    print("parameters:", n)

                    #create graph
                    I = tf.placeholder(tf.float64, shape = (n))
                    x,W,b = tf.split(I, [samples*inputs,units*inputs,units])
                    x = tf.reshape(x,(samples,inputs))
                    W = tf.reshape(W,(inputs,units))
                    b = tf.reshape(b,(1,units))

                    print("x:", x)
                    print("W:", W)
                    print("b:", b)

                    #new function
                    y = dense_module.dense(x, W, b)
                    #existing ops
                    #y = tf.matmul(x,W)+b     
                    #deadsimple
                    #y = tf.reduce_sum(x) + tf.reduce_sum(W) + tf.reduce_sum(b)

                    y = tf.reduce_sum(y)
                    print("y:", y)
                    
                    dydI = tf.gradients(y,I)

                    print("dydI:", dydI)

                    testdata = np.random.rand(n).astype(np.float64)-0.5
                    epsilon = 10e-7
                    dydi = np.asarray(sess.run(dydI, feed_dict = {I: testdata})).flatten()
                    #print("Gradient analytically: ", dydi.shape)
                    dydi_approx = np.zeros((n,),np.float64)
                    for i in range(n):
                        tmp = testdata[i]
                        testdata[i] = tmp - epsilon
                        y0 = sess.run(y, feed_dict = {I: testdata})
                        testdata[i] = tmp + epsilon
                        y1 = sess.run(y, feed_dict = {I: testdata})
                        testdata[i] = tmp
                        y0 = y0.flatten()
                        y1 = y1.flatten()
                        
                        dydi_approx[i] = (y1-y0)/(2*epsilon)
                    
                    #print("dydi_approx:", dydi_approx.shape)

                    gradient_error = np.linalg.norm(dydi_approx-dydi)/(np.linalg .norm(dydi_approx)+np.linalg.norm(dydi)) 
                    
                    gradient_eval = "worry much, something is wrong"
                    if gradient_error < 10e-7:
                        gradient_eval = "great"
                    elif gradient_error < 10e-5:
                        gradient_eval = "doublecheck for bugs"
                    

                    print("Gradient error on %s is %e, Evaluation: %s"%(device, gradient_error, gradient_eval))
        
                
if __name__ == '__main__':
    unittest.main()