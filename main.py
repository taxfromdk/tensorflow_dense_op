import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import math
#import _dense_grad
import random
import math
dense_module = tf.load_op_library('build/libdense.so')
import _dense_grad

#build simple network


LEARNING_RATE = 0.001
BATCH_SIZE = 22
RANGE = 100


#create graph
print("Tensorflow version: ", tf.__version__)
tf.set_random_seed(42)
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, 1), name="x")

def full_layer(i, units, af, n):
    print("---------full layer", units)
    print("i:", i.get_shape())
    W = tf.get_variable(n+"_W",shape=(i.get_shape()[1], units), initializer=tf.contrib.layers.xavier_initializer())
    print("W:", W.get_shape())
    b = tf.get_variable(n+"_b",shape=(1,units), initializer=tf.contrib.layers.xavier_initializer())
    print("b:", b.get_shape())
    #r_ = tf.matmul(i,W)
    #print("r_:", r_.get_shape())
    #r = r_ + b
    r = dense_module.dense(i,W,b)
    if af != None:
        r = af(r)
    print("--->r:", r.get_shape())
    
    return r
 
l = X   

#l =  tf.layers.dense(l, units=48, activation=tf.nn.relu)
#l =  tf.layers.dense(l, units=48, activation=tf.nn.relu)
#l =  tf.layers.dense(l, units=48, activation=tf.nn.relu)
#l =  tf.layers.dense(l, units=1, activation=None)

l =  full_layer(l, 48, tf.nn.relu, "A")
print(l)
l =  full_layer(l, 48, tf.nn.relu, "B")
print(l)
l =  full_layer(l, 48, tf.nn.relu, "C")
print(l)
l =  full_layer(l, 48, tf.nn.relu, "D")
print(l)
l =  full_layer(l, 1, None, "E")
print(l)

response = l 
           
Y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

loss = tf.reduce_sum(tf.pow(response - Y,2))
    
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

session = tf.Session()        
session.run(tf.global_variables_initializer())

def blackbox_function(x):
    return 3.0 + math.sin(x*math.pi*2/RANGE)

#def getBatch(f, n):
#    batch = []
#    for _ in range(n):
#        while True:
#            x = RANGE*(random.random()*2-1)
#            print(x)
#            batch.append(([x], [f(x)]))
#            break
#    i, o = zip(*batch)
#    return list(i), list(o)


plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim([-3,3])
plt.xlim([-RANGE,RANGE])

line_target, = ax.plot([0], [1], 'g.')
line_nn, = ax.plot([0], [1], 'r-')
    
def itemize(d):
    return list(map(lambda _: [_],d))

while True:
    d_x = [[(random.random()*2-1)*RANGE] for _ in range(BATCH_SIZE) ]
    d_y = [[blackbox_function(x[0])] for x in d_x]
    [tr, l, r] = session.run(fetches=[train_step, loss, response],feed_dict={X: d_x, Y: d_y})
    print("Loss: ", l)
    
    
    dp_x = [x*2*RANGE/100.0 - RANGE for x in range(100)]
    dp_x_ = [[x] for x in dp_x]
    
    [r] = session.run(fetches=[response],feed_dict={X: dp_x_})
    
    dp_target = [ blackbox_function(_) for _ in dp_x]
    dp_nn = [_[0] for _ in r]

    line_target.set_data(dp_x,dp_target)
    line_nn.set_data(dp_x,dp_nn)
        
    fig.canvas.draw()



