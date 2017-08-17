#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

train_x=np.load('./twitter/idx_q.npy', mmap_mode='r')
train_y=np.load('./twitter/idx_a.npy', mmap_mode='r')
print(train_x.shape)
print(train_x[0])

batch_size=128
sequence_length=4
num_encoder_symbols=10
num_decoder_symbols=10
embedding_size=16
learning_rate=0.01
MAX_GRAD_NORM=5
hidden_size=16

encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])
decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])

logits=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length,num_decoder_symbols])
targets=tf.placeholder(dtype=tf.int32,shape=[batch_size,sequence_length])
weights=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length])

train_decoder_inputs=np.zeros(shape=[batch_size,sequence_length],dtype=np.int32)


train_weights=np.ones(shape=[batch_size,sequence_length],dtype=np.float32)

cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

def seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size):
	encoder_inputs = tf.unstack(encoder_inputs, axis=1)
	decoder_inputs = tf.unstack(decoder_inputs, axis=1)
	results,states=tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    encoder_inputs,
    decoder_inputs,
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    output_projection=None,
    feed_previous=False,
    dtype=None,
    scope=None
)
	return results

def get_loss(logits,targets,weights):
	loss=tf.contrib.seq2seq.sequence_loss(
		logits,
		targets=targets,
		weights=weights
	)
	return loss

results=seq2seq(encoder_inputs,decoder_inputs,cell,num_encoder_symbols,num_decoder_symbols,embedding_size)
logits=tf.stack(results,axis=0)
print(logits)
loss=get_loss(logits,targets,weights)
trainable_variables=tf.trainable_variables()
grads,_=tf.clip_by_global_norm(tf.gradients(loss,trainable_variables),MAX_GRAD_NORM)
optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

with tf.Session() as sess:
	count=0

	while(count<100):
		print("count:", count)
		for step in range(0,100):
			#print("step:",step)
			sess.run(tf.global_variables_initializer())
			train_encoder_inputs=train_x[step*batch_size:step*batch_size+batch_size,:]
			train_targets=train_y[step*batch_size:step*batch_size+batch_size,:]
			#results_value=sess.run(results,feed_dict={encoder_inputs:train_encoder_inputs,decoder_inputs:train_decoder_inputs})

			op = sess.run(train_op, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
			                                 weights: train_weights, decoder_inputs: train_decoder_inputs})
			step=step+1
			if(step%10==0):
				cost = sess.run(loss, feed_dict={encoder_inputs: train_encoder_inputs, targets: train_targets,
												 weights: train_weights, decoder_inputs: train_decoder_inputs})
				print(cost)
		count=count+1