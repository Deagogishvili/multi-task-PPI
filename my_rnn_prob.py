# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang
@editor: Henriette Capel 4 Feb 2021

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional


class BiRNN_IF(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output):
        super(BiRNN_IF, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)

    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)

        return ppi_predictions_prob

class BiRNN_S8(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ss8_output):
        super(BiRNN_S8, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ss8_layer = keras.layers.Dense(ss8_output)

    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss8_predictions = self.ss8_layer(x)

        return ss8_predictions

class BiRNN_S3(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ss3_output):
        super(BiRNN_S3, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ss3_layer = keras.layers.Dense(ss3_output)

    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ss3_predictions = self.ss3_layer(x)

        return ss3_predictions

class BiRNN_SA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 asa_output):
        super(BiRNN_SA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.asa_layer = keras.layers.Dense(asa_output)

    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        asa_predictions = self.asa_layer(x)

        return asa_predictions

class BiRNN_BU(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 buried_output):
        super(BiRNN_BU, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.buried_layer = keras.layers.Dense(buried_output)

    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        buried_predictions = self.buried_layer(x)

        return buried_predictions

class BiRNN_IFBU(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output):
        super(BiRNN_IFBU, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)

        return ppi_predictions_prob, buried_predictions

class BiRNN_IFS3(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, ss3_output):
        super(BiRNN_IFS3, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)

    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        ss3_predictions = self.ss3_layer(x)

        return ppi_predictions_prob, ss3_predictions

class BiRNN_IFS8(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, ss8_output):
        super(BiRNN_IFS8, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        ss8_predictions = self.ss8_layer(x)

        return ppi_predictions_prob, ss8_predictions

class BiRNN_IFSA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, asa_output):
        super(BiRNN_IFSA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        asa_predictions = self.asa_layer(x)

        return ppi_predictions_prob, asa_predictions

class BiRNN_IFPP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, phipsi_output):
        super(BiRNN_IFPP, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        phipsi_predictions = self.phipsi_layer(x)

        return ppi_predictions_prob, phipsi_predictions


class BiRNN_IFBUS3(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss3_output):
        super(BiRNN_IFBUS3, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss3_predictions = self.ss3_layer(x)

        return ppi_predictions_prob, buried_predictions, ss3_predictions

class BiRNN_IFBUS8(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss8_output):
        super(BiRNN_IFBUS8, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss8_predictions = self.ss8_layer(x)

        return ppi_predictions_prob, buried_predictions, ss8_predictions

class BiRNN_IFBUSA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, asa_output):
        super(BiRNN_IFBUSA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        asa_predictions = self.asa_layer(x)

        return ppi_predictions_prob, buried_predictions, asa_predictions

class BiRNN_IFBUPP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, phipsi_output):
        super(BiRNN_IFBUPP, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        phipsi_predictions = self.phipsi_layer(x)

        return ppi_predictions_prob, buried_predictions, phipsi_predictions

class BiRNN_IFS3S8(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, ss3_output, ss8_output):
        super(BiRNN_IFS3S8, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        ss3_predictions = self.ss3_layer(x)
        ss8_predictions = self.ss8_layer(x)

        return ppi_predictions_prob, ss3_predictions, ss8_predictions

class BiRNN_IFS3SA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, ss3_output, asa_output):
        super(BiRNN_IFS3SA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        ss3_predictions = self.ss3_layer(x)
        asa_predictions = self.asa_layer(x)

        return ppi_predictions_prob, ss3_predictions, asa_predictions

class BiRNN_IFS8SA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, ss8_output, asa_output):
        super(BiRNN_IFS8SA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        ss8_predictions = self.ss8_layer(x)
        asa_predictions = self.asa_layer(x)

        return ppi_predictions_prob, ss8_predictions, asa_predictions

class BiRNN_IFBUS3S8(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss3_output, ss8_output):
        super(BiRNN_IFBUS3S8, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss3_predictions = self.ss3_layer(x)
        ss8_predictions = self.ss8_layer(x)

        return ppi_predictions_prob, buried_predictions, ss3_predictions, ss8_predictions

class BiRNN_IFSAS3S8(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, asa_output, ss3_output, ss8_output):
        super(BiRNN_IFSAS3S8, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        asa_predictions = self.asa_layer(x)
        ss3_predictions = self.ss3_layer(x)
        ss8_predictions = self.ss8_layer(x)

        return ppi_predictions_prob, asa_predictions, ss3_predictions, ss8_predictions


class BiRNN_IFBUS3SA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss3_output, asa_output):
        super(BiRNN_IFBUS3SA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss3_predictions = self.ss3_layer(x)
        asa_predictions = self.asa_layer(x)

        return ppi_predictions_prob, buried_predictions, ss3_predictions, asa_predictions

class BiRNN_IFBUS8SA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss8_output, asa_output):
        super(BiRNN_IFBUS8SA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss8_predictions = self.ss8_layer(x)
        asa_predictions = self.asa_layer(x)

        return ppi_predictions_prob, buried_predictions, ss8_predictions, asa_predictions

class BiRNN_IFBUSAPP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, asa_output, phipsi_output):
        super(BiRNN_IFBUSAPP, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        asa_predictions = self.asa_layer(x)
        phipsi_predictions = self.phipsi_layer(x)

        return ppi_predictions_prob, buried_predictions, asa_predictions, phipsi_predictions

class BiRNN_IFBUS3S8SA(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss3_output, ss8_output, asa_output):
        super(BiRNN_IFBUS3S8SA, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.asa_layer = keras.layers.Dense(asa_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss3_predictions = self.ss3_layer(x)
        ss8_predictions = self.ss8_layer(x)
        asa_predictions =self.asa_layer(x)

        return ppi_predictions_prob, buried_predictions, ss3_predictions, ss8_predictions, asa_predictions

class BiRNN_IFBUS3SAPP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss3_output, asa_output, phipsi_output):
        super(BiRNN_IFBUS3SAPP, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss3_predictions = self.ss3_layer(x)
        asa_predictions = self.asa_layer(x)
        phipsi_predictions = self.phipsi_layer(x)

        return ppi_predictions_prob, buried_predictions, ss3_predictions, asa_predictions, phipsi_predictions

class BiRNN_IFBUS8SAPP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss8_output, asa_output, phipsi_output):
        super(BiRNN_IFBUS8SAPP, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss8_predictions = self.ss8_layer(x)
        asa_predictions = self.asa_layer(x)
        phipsi_predictions = self.phipsi_layer(x)

        return ppi_predictions_prob, buried_predictions, ss8_predictions, asa_predictions, phipsi_predictions

class BiRNN_IFBUS3S8SAPP(keras.Model):
    def __init__(self, num_layers, units, rate, \
                 ppi_output, buried_output, ss3_output, ss8_output, asa_output, phipsi_output):
        super(BiRNN_IFBUS3S8SAPP, self).__init__()

        self.num_layers = num_layers

        self.birnn_layers = [
            Bidirectional(LSTM(units, dropout=rate, return_sequences=True), merge_mode='concat')
            for _ in range(self.num_layers)]

        self.dropout = keras.layers.Dropout(rate)

        self.ppi_layer = keras.layers.Dense(ppi_output)
        self.buried_layer = keras.layers.Dense(buried_output)
        self.ss3_layer = keras.layers.Dense(ss3_output)
        self.ss8_layer = keras.layers.Dense(ss8_output)
        self.asa_layer = keras.layers.Dense(asa_output)
        self.phipsi_layer = keras.layers.Dense(phipsi_output)

        self.probabilities = tf.keras.layers.Activation('softmax', dtype=tf.float32)


    def call(self, x, x_mask, training):

        # x.shape (batch_size, max_seq_length, embeded_size)
        # x_mask.shape (batch_size, timesteps).
        x_mask = tf.math.logical_not(tf.math.equal(x_mask, 1))

        for i in range(self.num_layers):
            x = self.birnn_layers[i](x, mask=x_mask, training=training)

        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        # predictions.shape: (batch_size, input_seq_len, output_size)
        ppi_predictions = self.ppi_layer(x)
        ppi_predictions_prob = self.probabilities(ppi_predictions)
        buried_predictions = self.buried_layer(x)
        ss3_predictions = self.ss3_layer(x)
        ss8_predictions = self.ss8_layer(x)
        asa_predictions =self.asa_layer(x)
        phipsi_predictions = self.phipsi_layer(x)

        return ppi_predictions_prob, buried_predictions, ss3_predictions, ss8_predictions, asa_predictions, phipsi_predictions
