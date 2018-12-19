# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import time
import os
import Siamese_Net
# Importing dataset
# 123
class Dataset:
    """
    Store data and provide interface for batches training and testing
    Create positive and negative pairs with ratio 1:1
    Ratio of pairs per lavel depends on labels_equal

    Requires: data, labels (with one hot encoding), number of labels, (eqal)
    """
    def __init__(self, data, labels, n_labels, labels_equal=False, max_pairs=-1):
        self.labels = labels
        self.n_labels = n_labels
        self.label_indices = [np.where(np.argmax(labels, 1) == i)[0]
                              for i in range(n_labels)]
        self.data = data
        self.epoch = 0
        self.labels_equal = labels_equal
        self.max_pairs = max_pairs
        self.pos_pairs, self.pos_label_pairs = self.generatePosPairs()
        self.neg_pairs, self.neg_label_pairs = self.generateNegPairs()
        self.length = len(self.pos_pairs)
        self.index = 0

    def generatePosPairs(self):
        """ Returns positive pairs created from data set """
        pairs = []
        label_pairs = []
        labels_len = [len(self.label_indices[d])
                      for d in range(self.n_labels)]

        start_time = time.time() # DEBUG

        if self.labels_equal or self.max_pairs != -1:
            # Number of pairs depends on smallest label dataset
            n = min(labels_len)

            lab = 0
            idx = 0
            pad = 1

            while len(pairs) < self.max_pairs and pad < n:
                pairs += [[self.data[self.label_indices[lab][idx]],
                           self.data[self.label_indices[lab][idx + pad]]]]
                label_pairs += [[self.labels[self.label_indices[lab][idx]],
                           self.labels[self.label_indices[lab][idx + pad]]]]

                lab = (lab + 1) % self.n_labels
                if lab == 0:
                    idx += 1
                    if (idx + pad) >= n:
                        idx = 0
                        pad += 1

        else:
            # Create maximum number of pairs
            for lab in range(self.n_labels):
                n = labels_len[lab]
                for i in range(n-1):
                    for ii in range(i+1, n):
                        pairs += [[self.data[self.label_indices[lab][i]],
                                    self.data[self.label_indices[lab][ii]]]]
                        label_pairs += [[self.labels[self.label_indices[lab][i]],
                                         self.labels[self.label_indices[lab][ii]]]]

        print("Positive pairs generated in", time.time() - start_time) # DEBUG
        return np.array(pairs), np.array(label_pairs)

    def generateNegPairs(self):
        """ Retruns random negative pairs same length as positive pairs """
        pairs = []
        label_pairs = []
        chosen = []
        i = 0
        start_time = time.time() # DEBUG
        while len(pairs) < len(self.pos_pairs):
            ii = (i + random.randrange(1, self.n_labels)) % self.n_labels
            choice = [random.choice(self.label_indices[i]),
                      random.choice(self.label_indices[ii])]
            if choice not in chosen:
                chosen += [choice]
                pairs += [[self.data[choice[0]], self.data[choice[1]]]]
                label_pairs += [[self.labels[choice[0]], self.labels[choice[1]]]]
            i = (i + 1) % self.n_labels

        print("Negative pairs generated in", time.time() - start_time) # DEBUG
        return np.array(pairs), np.array(label_pairs)

    def get_epoch(self):
        """ Get current dataset epoch """
        return self.epoch

    def get_length(self):
        """ Get positive pairs length """
        return self.length

    def next_batch(self, batch_size):
        """
        Returns batch of images and labels of given length
        Requires: even batch size
        """
        start = self.index
        l_size = int(batch_size / 2)
        self.index += l_size

        if self.index > self.length:
            # Shuffle the data
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.pos_pairs = self.pos_pairs[perm]
            self.pos_label_pairs = self.pos_label_pairs[perm]
            self.neg_pairs, self.neg_label_pairs = self.generateNegPairs()
            # Start next epoch
            start = 0
            self.epoch += 1
            self.index = l_size

        end = self.index
        return (np.append(self.pos_pairs[start:end],
                          self.neg_pairs[start:end], 0),
                np.append(np.ones((l_size, 1)),
                          np.zeros((l_size, 1)), 0),
                np.append(self.pos_label_pairs[start:end],
                          self.neg_label_pairs[start:end], 0))

    def random_batch(self, batch_size):
        """
        Returns random randomly shuffled batch - for testing
        *** Maybe not neccesary ***
        """
        pass


# Layers for CNN
X_train = np.load('mstar_train_image.npy').reshape([-1, 4096])
y_train = np.load('mstar_train_label_1.npy')
X_test = np.load('mstar_test_image.npy').reshape([-1,4096])
y_test = np.load('mstar_test_label_1.npy')

tr_data = Dataset(X_train, y_train, 10, max_pairs= -1)
te_data = Dataset(X_test, y_test, 10, max_pairs= -1)


### MODEL
images_L = tf.placeholder(tf.float32,shape=([None,4096]),name='images_L')
images_R = tf.placeholder(tf.float32,shape=([None,4096]),name='images_R')
labels = tf.placeholder(tf.float32,shape=([None,1]), name='labels')
labels_L = tf.placeholder(tf.float32,shape=([None,10]), name='labels_L')
labels_R = tf.placeholder(tf.float32,shape=([None,10]), name='labels_R')


with tf.variable_scope("ConvSiameseNet") as scope:
    model_L = Siamese_Net.convnet(images_L)
    scope.reuse_variables()
    model_R = Siamese_Net.convnet(images_R)

# Combine two outputs by L1 distance
distance_L1 = tf.abs(tf.subtract(model_L, model_R))
# distance_L2 = tf.sqrt(tf.reduce_sum(tf.square(model_L - model_R), axis=1))
# margin = tf.constant(5.0)
# contastive_distance = tf.multiply(labels, tf.square(distance_L2)) + tf.multiply(tf.subtract(1, labels), tf.square( tf.maximum( tf.subtract(margin, distance_L2), 0 )))
# Final layer with sigmoid
W_out = tf.get_variable('W_out', shape=[1024, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.Variable(tf.constant(0.1, shape=[1]), name="b_out")
# Output - result of sigmoid - for future use
# Prediction - rounded sigmoid to 0 or 1
output = tf.nn.sigmoid(tf.matmul(distance_L1, W_out) + b_out)
prediction = tf.round(tf.subtract(output,0.1))
# Using cross entropy for sigmoid as loss
# @TODO add regularization
loss_Dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output))

# classifier loss
with tf.name_scope('L_cls_layer'):
    W_cls = tf.Variable(initial_value=tf.random_normal(shape=[1024, 10], stddev=0.01), name='cls_Weights')
    b_cls = tf.Variable(initial_value=tf.zeros(shape=[10]), name='cls_bias')
    logits_L = tf.matmul(model_L, W_cls) + b_cls
    L_pred = tf.nn.softmax(logits=logits_L)
    # 求交叉熵损失
    L_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_L, logits=L_pred, name='cls_cross_entropy')
    # 求平均
    loss_cls_L = tf.reduce_mean(L_cross_entropy, name='loss')

    logits_R = tf.matmul(model_R, W_cls) + b_cls
    R_pred = tf.nn.softmax(logits=logits_R)
    # 求交叉熵损失
    R_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_R, logits=R_pred, name='cls_cross_entropy')
    # 求平均
    loss_cls_R = tf.reduce_mean(R_cross_entropy, name='loss')

# Total Loss
loss = loss_Dis + ( loss_cls_L + loss_cls_R )* 0.1
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00005).minimize(loss)
# Measuring accuracy of model by distance
#correct_prediction = tf.equal(prediction, labels)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.name_scope('Softmax_Evaluate'):
    # 返回验证集/测试集预测正确或错误的布尔值
    correct_prediction = tf.equal(tf.argmax(L_pred, 1), tf.argmax(labels_L, 1))
    # 将布尔值转换为浮点数后，求平均准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



### TRAINING
batch_size = 64 # 128

with tf.Session() as sess:
    print("Starting training")
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    model_ckpt = 'model.ckpt.data-00000-of-00001'
    if os.path.isfile(model_ckpt):
        saver.restore(sess, 'model.ckpt')
        print('true')
    
    # Training cycle
    for epoch in range(181):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = 150            # Not accurate
        start_time = time.time()

        # Loop over all batches
        for i in range(total_batch):
            # Fit training using batch data
            tr_input, y, real_y = tr_data.next_batch(batch_size)
            _, loss_value, acc, pre = sess.run([optimizer, loss, accuracy, prediction],
                                          feed_dict={images_L: tr_input[:,0],
                                                     images_R: tr_input[:,1],
                                                     labels_L: real_y[:, 0],
                                                     labels_R: real_y[:, 1],
                                                     labels: y})
            avg_loss += loss_value
            avg_acc += acc * 100

        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' % (epoch,
                                                           duration,
                                                           avg_loss/total_batch,
                                                           avg_acc/total_batch))

        #print(pre)
        te_pairs, te_y, real_te_y = te_data.next_batch(1000)
        te_acc = accuracy.eval(feed_dict={images_L: te_pairs[:,0],
                                          images_R: te_pairs[:,1],
                                          labels_L: real_te_y[:, 0],
                                          labels_R: real_te_y[:, 1],
                                          labels: te_y})
        print('Accuracy on test set %0.2f' % (100 * te_acc))
        if (epoch % 10 == 0):
            saver.save(sess, './model.ckpt')


    # Final Testing
    tr_pairs, tr_y, real_tr_y = te_data.next_batch(1000)
    tr_acc = accuracy.eval(feed_dict={images_L: tr_pairs[:,0],
                                      images_R: tr_pairs[:,1],
                                      labels_L: real_tr_y[:, 0],
                                      labels_R: real_tr_y[:, 1],
                                      labels: tr_y})
    print('Accuract training set %0.2f' % (100 * tr_acc))

    te_pairs, te_y, finalReal_te_y = te_data.next_batch(1000)
    te_acc = accuracy.eval(feed_dict={images_L: te_pairs[:,0],
                                      images_R: te_pairs[:,1],
                                      labels_L: finalReal_te_y[:, 0],
                                      labels_R: finalReal_te_y[:, 1],
                                      labels: te_y})
    print('Accuract test set %0.2f' % (100 * te_acc))


    # TODO Predicting correct label based accuracy on sample of labeled data.
    # test2label
    # right_pred_number = 0
    # for k in range(600):
    #     sum = np.zeros([10, ])
    #     for i in range(240):
    #         pred = prediction.eval(feed_dict={images_L:[X_test[k,:]],
    #                                           images_R:[X_train[i,:]]
    #         })
    #         train_data_label = int(i/24)
    #         if pred ==1 :
    #             sum[train_data_label] += 1
    #         else:
    #             sum[train_data_label] -= 1
    #     pred_label = np.argmax(sum)
    #     print ('%d : %d' %(k,pred_label))
    #
    #     #test this image if clsed right
    #     test_data_label = k//60
    #     if pred_label == test_data_label:
    #         right_pred_number += 1
    #
    # print(right_pred_number/600.00)
    total_acc = accuracy.eval(feed_dict={images_L: X_test,
                                      labels_L: y_test})
    print('Accuract test set %0.2f' % (100 * total_acc))


