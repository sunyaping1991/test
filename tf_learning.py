#coding :utf-8
import tensorflow as tf
import numpy as np
from  tensorflow.examples.tutorials.mnist import   input_data
# mnist=input_data.read_data_sets(".",one_hot=True)
#
# x = tf.placeholder('float',[None,784])
# w  = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x,w)+b)
# y_ = tf.placeholder("float",[None,10])
#
# cross_entropy=-tf.reduce_sum(y_ * tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# init  = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# for i in range(3000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     # print(batch_ys)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#     if i%100 == 0:
#         # 评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置
#         correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#         print("第" + str(i) + "步骤为:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]  #300*1,输入只有一个神经元
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

prediction = add_layer(l1,10,1,activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss=loss)

init  = tf.global_variables_initializer()
import matplotlib.pyplot as plt

loss_list = []
with  tf.Session()  as sess:
    sess.run(init)
    for i in range(5000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 100 == 0:
            ss = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            print(ss)
            loss_list.append(ss)
plt.plot([x+1 for x in range(len(loss_list))],loss_list)
plt.show()
