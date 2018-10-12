import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOAD_MODEL = False
LOAD_MODEL_PATH = './vae/model'
BATCH_SIZE = 32
EPOCH = 50000
SAVE_DIR = '/home/konosuke-a/python_code/cnncancer_k/MNIST/vae'   # absolute path
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

def batchnorm(input):
    with tf.variable_scope('batchnorm',reuse=tf.AUTO_REUSE):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        print(input.get_shape())
        offset = tf.get_variable("offset_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale_{}".format(channels), [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

def Save_all_variables(save_dir,sess):
    names = [v.name for v in tf.trainable_variables()]
    for n in names:
        v = tf.get_default_graph().get_tensor_by_name(n)
        save_name = n.split('/')[-1].split(':')[0]
        print('saved variables  ', save_dir +'/variables/'+ save_name + '.npy')
        np.save(save_dir +'/variables/'+ save_name + '.npy', sess.run(v))


def lrelu(x,a=0.2):
    with tf.name_scope('lrelu'):
        x = tf.identity(x)
        return (0.5 * (1+a)*x + (0.5*(1-a)) * tf.abs(x))

with tf.variable_scope('Encoder'):
    with tf.variable_scope('Input'):
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 784], name='input_x')
#        x_image = tf.reshape(x,[-1,28,28,1])
    
    with tf.variable_scope('layer_1'):
        # [n,28,28,1] -> [n,14,14,32]
        w1 = tf.get_variable(name='w1',shape=[784,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b1 = tf.get_variable(name='b1',shape=[1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        y1 = tf.nn.tanh(tf.matmul(x,w1) + b1)
    
    with tf.variable_scope('layer_2'):
        w2 = tf.get_variable(name='w2',shape=[1000,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b2 = tf.get_variable(name='b2',shape=[1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        y2 = tf.nn.tanh(tf.matmul(y1,w2) + b2)
    
    with tf.variable_scope('mu_vector'):
        w_mu = tf.get_variable(name='w_mu',shape=[1000,2], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b_mu = tf.get_variable(name='b_mu',shape=[2], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        y_mu = tf.matmul(y2,w_mu) + b_mu
    
    with tf.variable_scope('sigma_vector'):
        w_sigma = tf.get_variable(name='w_sigma',shape=[1000,2], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b_sigma = tf.get_variable(name='b_sigma',shape=[2], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        y_sigma = tf.matmul(y2,w_sigma) + b_sigma
    
    with tf.variable_scope('z_vector'):
        z_vector = y_mu + y_sigma*tf.random_normal([BATCH_SIZE,2],mean=0,stddev=1)

with tf.variable_scope('Decoder'):
    with tf.variable_scope('FC'):
        w_fc = tf.get_variable(name='w_fc', shape=[2,1000])
        b_fc = tf.get_variable(name='b_fc', shape=[1000])
        fc = tf.nn.relu(tf.matmul(z_vector,w_fc) + b_fc)
    
    with tf.variable_scope('layer_2t'):
        w2t = tf.get_variable(name='w2t',shape=[1000,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b2t = tf.get_variable(name='b2t',shape=[1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        y2t = tf.nn.relu(tf.matmul(fc,w2t) + b2t)

    with tf.variable_scope('layer_1t'):
        w1t = tf.get_variable(name='w1t',shape=[1000,1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        b1t = tf.get_variable(name='b1t',shape=[1000], dtype=tf.float32, initializer=tf.random_normal_initializer(0,0.02))
        y1t = tf.nn.sigmoid(tf.matmul(y2t,w1t) + b1t)

with tf.variable_scope('loss'):
    EPS = 1e-7
    # NOTE なぜか論文記載の方法では学習がうまくいかなかった. 後ほど原因検証.
    #reconsruct_loss = 1/BATCH_SIZE * tf.reduce_sum(x_image*tf.log(deconv1+EPS) - (1-x_image)*tf.log(1-deconv1+EPS))
    #kl_loss = 0.5*tf.reduce_sum(1+tf.log(sigma3**2 + EPS) - mu3**2 - sigma3**2)
    #reconsruct_loss = -tf.reduce_sum(x_image*tf.log(tf.clip_by_value(deconv1,1e-20,1e+20)) + (1-x_image)*tf.log(tf.clip_by_value(1-deconv1,1e-20,1e+20)))
    reconsruct_loss = -tf.reduce_sum(x*tf.log(y1t + EPS) + (1-x)*tf.log(1-y1t + EPS))
    #reconsruct_loss = tf.reduce_sum(tf.square(x_image-deconv1))/BATCH_SIZE

    kl_loss = 0.5*tf.reduce_sum(tf.square(y_mu) + tf.exp(y_sigma)**2 - 2*y_sigma -1)
    loss = reconsruct_loss + kl_loss

with tf.variable_scope('optimize'):
    trainable_vars_list = [var for var in tf.trainable_variables()]
    adam = tf.train.AdamOptimizer(0.0002,0.5)
    gradients_vars = adam.compute_gradients(loss, var_list=trainable_vars_list)
    train_op = adam.apply_gradients(gradients_vars)

with tf.name_scope('summary'):
    with tf.name_scope('Input_image_summary'):
        tf.summary.image('Input_image', tf.image.convert_image_dtype(tf.reshape(x,[BATCH_SIZE,28,28,1]), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Reconstruct_image_summary'):
        tf.summary.image('Reconstruct_image', tf.image.convert_image_dtype(tf.reshape(y1t,[BATCH_SIZE,28,28,1]), dtype=tf.uint8, saturate=True))

    with tf.name_scope('Loss_summary'):
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('reconsruct_loss', reconsruct_loss)
        tf.summary.scalar('kl_loss', kl_loss)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/Variable_histogram', var)

    for grad, var in gradients_vars:
        tf.summary.histogram(var.op.name + '/Gradients', grad)

# Session
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    if LOAD_MODEL is not True:
        sess.run(init)
    
        # mkdir if not exist directory
        if not os.path.exists(SAVE_DIR): # NOT CHANGE
            os.mkdir(SAVE_DIR)
            os.mkdir(os.path.join(SAVE_DIR,'summary'))
            os.mkdir(os.path.join(SAVE_DIR,'variables'))
            os.mkdir(os.path.join(SAVE_DIR,'model'))
        
        # remove old summary if already exist
        if tf.gfile.Exists(os.path.join(SAVE_DIR,'summary')):    # NOT CHANGE
            tf.gfile.DeleteRecursively(os.path.join(SAVE_DIR,'summary'))
        
        # merging summary & set summary writer
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR,'summary'), graph=sess.graph)
    
        saver = tf.train.Saver()
        # train
        for step in range(EPOCH):
            x_batch, t_batch = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x:x_batch}) 
            if step % 50 == 0:
                print('step', step)
                print('reconsruct_loss: ', sess.run(reconsruct_loss, feed_dict={x:x_batch}))
                print('kl_loss: ', sess.run(kl_loss, feed_dict={x:x_batch}))
                print('total_loss: ', sess.run(loss, feed_dict={x:x_batch}))
                print()
                summary_writer.add_summary(sess.run(merged, feed_dict={x:x_batch}), step)
        
        Save_all_variables(save_dir=SAVE_DIR,sess=sess)
        saver.save(sess, SAVE_DIR + "/model/model.ckpt")   

    else:
        saver.restore(sess, LOAD_MODEL_PATH + '/model.ckpt')
        print('Load model from {}'.format(LOAD_MODEL_PATH))

    # test (plot latent z-vector)
    x_test, t_test = mnist.test.images, mnist.test.labels
    batches = np.arange(0,10000-32,32)
    z_tmp = np.zeros([np.max(batches)+32, 2])
    for i in batches:
        z_tmp[i:i+32] = sess.run(z_vector, feed_dict={x:x_test[i:i+32]})
    
    plt.figure()
    plt.title("Plot latent vector of VAE")
    for i in range(len(z_tmp)):
        if t_test[i] == 0:
            p0 = plt.scatter(z_tmp[i,0], z_tmp[i, 1],c="red",s=1)
        if t_test[i] == 1:
            p1 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="blue",s=1)
        if t_test[i] == 2:
            p2 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="green",s=1)
        if t_test[i] == 3:
            p3 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="pink",s=1)
        if t_test[i] == 4:
            p4 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="yellow",s=1)
        if t_test[i] == 5:
            p5 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="orange",s=1)
        if t_test[i] == 6:
            p6 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="cyan",s=1)
        if t_test[i] == 7:
            p7 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="deeppink",s=1)
        if t_test[i] == 8:
            p8 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="c",s=1)
        if t_test[i] == 9:
            p9 = plt.scatter(z_tmp[i,0], z_tmp[i, 1], c="purple",s=1)
    plt.legend([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9],["0","1","2","3","4","5","6","7","8","9"], bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
    filename = "VAE_latent_ver2.png"
    plt.savefig(filename)
    
    # TODO 10dim 混合ガウス分布を用いて、
    # unsupervisedに論文と同様の結果が得られるのか検証
    # ckptでモデル保存できるようにしたほうが簡単かも
