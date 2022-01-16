#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:47:02 2018

@author: flyaway
"""

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import numpy as np
import keras
import tensorflow as tf

class Attention_layer(Layer):
    def __init__(self,
                 W_regularizer=None,
                 U_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 U_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        super(Attention_layer, self).__init__(**kwargs)
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.U_constraint = constraints.get(U_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        self.supports_masking = True
    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.U = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        # add_weight()函数中的trainable默认为True,所以已经将W,b变成可以训练的参数了
        super(Attention_layer, self).build(input_shape)
    def compute_mask(self, inputs, mask=None):
        return None
    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = K.dot(uit, self.U)
        a = K.exp(ait)
        if mask is not None:
            # cast the mask to floatx to avoid float64 upcasting int theano
            out_mask = K.reshape(mask, (mask.shape[0], 1))
            a *= K.cast(out_mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # keepdims = True保持维度，如输入是2D,输出也是2D
        # keepdims = False,输入2D,输出1D
        # print a
        # print x
        weighted_input = x * a  # a the same dim as x ,这里的×是点乘
        print(a)
        # print weighted_input
        return K.sum(weighted_input, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# 该类未实现
class Direction_Attention_layer(Layer):
    def __init__(self,
                 W_regularizer=None,
                 U_regularizer=None,
                 b_regularizer=None,
                 train_batchsize=None,
                 test_batchsize=None,
                 istrain=True,
                 W_constraint=None,
                 U_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):

        super(Direction_Attention_layer, self).__init__(**kwargs)
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.U_constraint = constraints.get(U_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.istrain = istrain
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        if self.istrain == True:
            bs = self.train_batchsize
        else:
            bs = self.test_batchsize
        print("bs: ", bs)

        sl = input_shape[1]
        sl_indices = np.array(range(sl))
        sl_col, sl_row = np.meshgrid(sl_indices, sl_indices)

        self.fw_mask = K.cast(K.cast(K.greater(sl_row, sl_col), tf.int32), tf.float32)
        self.bw_mask = K.cast(K.cast(K.greater(sl_col, sl_row), tf.int32), tf.float32)

        # diag_mask = K.cast(np.diag(-np.ones([sl],np.int32)) + 1,np.bool)
        # self.fw_mask_tile = tf.tile(K.expand_dims(fw_mask, 0), [bs, 1, 1])
        # self.bw_mask_tile = tf.tile(K.expand_dims(bw_mask, 0), [bs, 1, 1])
        # self.fw_mask_tile = K.cast(K.cast(K.tile(K.expand_dims(fw_mask, 0), [bs, 1, 1]), tf.int32),tf.float32) # bs,sl,sl
        # self.bw_mask_tile = K.cast(K.cast(K.tile(K.expand_dims(bw_mask, 0), [bs, 1, 1]), tf.int32),tf.float32)

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.U = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        # add_weight()函数中的trainable默认为True,所以已经将W,b变成可以训练的参数了

        super(Direction_Attention_layer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        ait = K.dot(uit, self.U)

        a = K.exp(ait)

        if mask is not None:
            out_mask = K.reshape(mask, (mask.shape[0], 1))
            a *= K.cast(out_mask, K.floatx())

        fw_a = tf.multiply(a, self.fw_mask)
        bw_a = tf.multiply(a, self.bw_mask)

        # fw_a = keras.layers.Multiply()([a, self.fw_mask_tile])
        # bw_a = keras.layers.Multiply()([a, self.bw_mask_tile])

        # a /= K.cast(K.sum(a,axis = 1,keepdims = True) + K.epsilon(),K.floatx())
        fw_a = K.cast(K.sum(fw_a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        bw_a = K.cast(K.sum(bw_a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input_fw = K.sum(x * fw_a, axis=1)  # a the same dim as x ,这里的×是点乘
        weighted_input_bw = K.sum(x * bw_a, axis=1)

        res = tf.concat([weighted_input_fw, weighted_input_bw], axis=-1)
        print("res.shape", res.shape)
        return res

        # return K.sum(weighted_input,axis = 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * input_shape[-1])


class Mult_Attention_layer(Layer):
    def __init__(self,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True, **kwargs):

        super(Mult_Attention_layer, self).__init__(**kwargs)
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        # add_weight()函数中的trainable默认为True,所以已经将W,b变成可以训练的参数了
        super(Mult_Attention_layer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        a = K.dot(x, self.W)

        if self.bias:
            a += self.b

        # uit = K.tanh(uit)

        # ait = K.dot(uit,self.U)

        # a = K.exp(ait)

        if mask is not None:
            # cast the mask to floatx to avoid float64 upcasting int theano
            out_mask = K.reshape(mask, (mask.shape[0], 1))
            a *= K.cast(out_mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # keepdims = True保持维度，如输入是2D,输出也是2D
        # keepdims = False,输入2D,输出1D

        # print a
        # print x

        weighted_input = x * a  # a the same dim as x ,这里的×是点乘
        print(a)
        # print weighted_input

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
