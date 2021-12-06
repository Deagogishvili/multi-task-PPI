# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang
@editor: Henriette Capel (05-02-2021)

"""

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
# import keras
# from keras import backend as K

def read_filenames(data_list):

    filenames = []
    f = open(data_list, 'r')
    for i in f.readlines():
        if i.strip() != "":
            filenames.append(i.strip())
    f.close()

    return filenames

ratio = 0.25

def get_enhancement(inputs, index=None):

    if index != None:
        return inputs[index[0]:index[1]]
    else:
        length = inputs.shape[0]
        # about half
        if np.random.randint(0,2) == 0:
            return inputs, [0, length]
        else:
            start = np.random.randint(0, int(length*ratio))
            end = length - np.random.randint(0, int(length*ratio))
            return inputs[start:end], [start, end]

def read_inputs(filenames, inputs_files_path, data_enhance, input_norm):
    """
    20pssm + 30hhm + 7pc + 19psp
    """
    inputs_nopadding = []
    max_len = 0
    inputs_total_len = 0
    indices = []
    for filename in filenames:
        inputs_ = np.loadtxt((os.path.join(inputs_files_path, filename + ".inputs")))

        if data_enhance:
            inputs_, index = get_enhancement(inputs_)
            indices.append(index)

        inputs_total_len += inputs_.shape[0]
        if inputs_.shape[0] > max_len:
            max_len = inputs_.shape[0]
        inputs_nopadding.append(inputs_)

    inputs_padding = np.zeros(shape=(len(filenames), max_len, 76))
    inputs_mask_padding = np.ones(shape=(len(filenames), max_len))

    for i in range(len(filenames)):
        inputs_padding[i,:inputs_nopadding[i].shape[0]] = inputs_nopadding[i]
        inputs_mask_padding[i,:inputs_nopadding[i].shape[0]] = 0

    if input_norm:
        #(hhm - 5000) / 1000
        inputs_padding[:,:,20:50] = (inputs_padding[:,:,20:50] - 5000)/1000

    return inputs_padding, inputs_mask_padding, inputs_total_len, indices

def read_labels(filenames, labels_files_path, fastas_files_path, data_enhance, indices, part_ppi_list):
    """
    8ss(one-hot) + 3csf(double) + [2*(phi+psi) + 2*(x1+x2+x3+x4)](sin,cos) + asa + ss3 + buried + ppi
    8 + 3 + 4 + 8 + 1

    ss_labels = labels[:,:,:8]
    csf_labels = labels[:,:,8:11]
    phipsi_labels = labels[:,:,11:15]
    dihedrals_labels = labels[:,:,15:23]
    asa_labels = labels[:,:,23]
    real_phipsidihedrals=labels[:,:,24:30]
    ss3 = labels[:,:,30:33]
    buried = labels[:,:,33]
    nonburied = labels[:,:,34]
    ppi = labels[:,:,35]

    """
    labels_nopadding = []
    masks_nopadding = []
    max_len = 0
    labels_total_len = 0
    for idx, filename in enumerate(filenames):
        labels_ = np.loadtxt((os.path.join(labels_files_path, filename + ".labels")))
        masks_ = np.loadtxt((os.path.join(labels_files_path, filename + ".labels_mask")))
        fasta_ = open(os.path.join(fastas_files_path, filename + ".fasta"), "r")

        #last three csf
        masks_[-3:,8:11] = 1

        # mask PPI annotations that are not present in part_ppi_list
        if filename not in part_ppi_list:
            masks_[:,-1] = 1

        if data_enhance:
            labels_ = get_enhancement(labels_, indices[idx])
            masks_ = get_enhancement(masks_, indices[idx])

        assert labels_.shape[0] == masks_.shape[0]
        labels_total_len += labels_.shape[0]
        if labels_.shape[0] > max_len:
            max_len = labels_.shape[0]
        labels_nopadding.append(labels_)
        masks_nopadding.append(masks_)

        num_columns_labels = labels_.shape[1]


    #Note, the last dimension was set to 35. Why?
    labels_padding = np.zeros(shape=(len(filenames), max_len, num_columns_labels))
    masks_padding = np.ones(shape=(len(filenames), max_len, num_columns_labels))

    for i in range(len(filenames)):
        labels_padding[i,:labels_nopadding[i].shape[0]] = labels_nopadding[i]
        masks_padding[i,:masks_nopadding[i].shape[0]] = masks_nopadding[i]

    return labels_padding, masks_padding, labels_total_len

class InputReader(object):

    def __init__(self, data_list, inputs_files_path, labels_files_path, fastas_files_path,\
                 num_batch_size, ppi_list = False, part_ppi = False, input_norm=False, shuffle=False, data_enhance=False):

        self.filenames = read_filenames(data_list)
        if ppi_list:
            self.filenames_ppi = read_filenames(ppi_list)

        self.inputs_files_path = inputs_files_path
        self.labels_files_path = labels_files_path
        self.fastas_files_path = fastas_files_path
        self.input_norm = input_norm
        self.data_enhance = data_enhance

        if part_ppi:
            if ppi_list:       #FOR TRAINING TAKE PART PPI
                tf.print("sequences in filenames")
                tf.print(len(self.filenames))
                tf.print("sequences in filenames_ppi")
                tf.print(len(self.filenames_ppi))
                self.part_ppi_list = self.filenames_ppi[::part_ppi]
                tf.print("sequences in part_ppi_list")
                tf.print(len(self.part_ppi_list))
            else:                       #FOR VALIDATION/TEST TAKE ALL DATA
                tf.print("sequences in filenames")
                tf.print(len(self.filenames))
                self.part_ppi_list = self.filenames[::part_ppi]
                tf.print("sequences in part_ppi_list")
                tf.print(len(self.part_ppi_list))

        if self.data_enhance:
            print ("use data enhancement...")

        if shuffle:
            self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames) \
                .shuffle(len(self.filenames)).batch(num_batch_size)
        else:
             self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames) \
                .batch(num_batch_size)

        print ("Data Size:", len(self.filenames))

    def read_file_from_disk(self, filenames_batch):

        filenames_batch = [bytes.decode(i) for i in filenames_batch.numpy()]
        inputs_batch, inputs_masks_batch, inputs_total_len, indices = \
            read_inputs(filenames_batch, self.inputs_files_path, self.data_enhance, self.input_norm)
        labels_batch, labels_masks_batch, labels_total_len = \
            read_labels(filenames_batch, self.labels_files_path, self.fastas_files_path, self.data_enhance, indices, self.part_ppi_list)

        inputs_batch = tf.convert_to_tensor(inputs_batch, dtype=tf.float32)
        inputs_masks_batch= tf.convert_to_tensor(inputs_masks_batch, dtype=tf.float32)
        labels_batch = tf.convert_to_tensor(labels_batch, dtype=tf.float32)
        labels_masks_batch= tf.convert_to_tensor(labels_masks_batch, dtype=tf.float32)

        return filenames_batch, inputs_batch, inputs_masks_batch, \
            labels_batch, labels_masks_batch, inputs_total_len, labels_total_len

cross_entropy_loss_func = keras.losses.CategoricalCrossentropy(
    reduction = keras.losses.Reduction.NONE, from_logits=True)

def loss_function(real, pred, loss_weights):
    loss_ = cross_entropy_loss_func(real, pred)
    loss_weights = tf.cast(loss_weights, dtype=loss_.dtype)
    loss_ *= loss_weights
    loss_ =  tf.reduce_sum(loss_)/tf.reduce_sum(loss_weights)

    return loss_


def compute_cross_entropy_loss(predictions, labels, labels_mask):

    # labels.shape: batch, seq_len, 8
    # labels_mask.shape: batch, seq_len, 8
    # predictions.shape: batch, seq_len, 8

    labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
    labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
    predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

    # labels_mask.shape: batch, seq_len
    labels_mask = labels_mask[:,0]
    indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
    labels_ = tf.gather(labels, indices)
    predictions_ = tf.gather(predictions, indices)

    # loss_.shape: batch*seq_len, 8
    loss_ = cross_entropy_loss_func(labels_, predictions_)
    loss_ = tf.reduce_mean(loss_)

    return loss_

def compute_cross_entropy_loss_ppi(predictions, labels, labels_mask, class_weights):

    labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
    labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
    predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

    #Note: labels have different size now
    loss_weigth = tf.ones([tf.shape(labels)[0], tf.shape(labels)[1]], tf.float32)

    # labels_mask.shape: batch, seq_len
    labels_mask = labels_mask[:,0]
    indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
    labels_ = tf.gather(labels, indices)
    predictions_ = tf.gather(predictions, indices)
    loss_weigth_ = tf.gather(loss_weigth, indices)

    labels_NIF = tf.zeros([tf.shape(labels_)[0], tf.shape(labels_)[1]], tf.float32)
    labels_NIF = tf.where(tf.math.not_equal(labels_, 0), labels_NIF, [1])
    labels_2dim = tf.concat([labels_NIF, labels_], 1)

    loss_weigth_ = tf.where(tf.math.not_equal(labels_NIF, 1), loss_weigth_, [class_weights[0]])
    loss_weigth_ = tf.where(tf.math.not_equal(labels_NIF, 0), loss_weigth_, [class_weights[1]])
    #labels_ and predictions_ shape: residues * 2  --> as should be [batch_size, num_classes]

    loss_weight_1D = tf.reshape(loss_weigth_, [tf.shape(loss_weigth_)[0]])

    loss_ = cross_entropy_loss_func(labels_2dim, predictions_, sample_weight= loss_weight_1D)
    loss_ = tf.reduce_mean(loss_)

    return loss_

mse_loss_func = keras.losses.MeanSquaredError()

def compute_mse_loss(predictions, labels, labels_mask):

    # labels.shape: batch, seq_len, 4
    # labels_mask.shape: batch, seq_len, 4
    # predictions.shape: batch, seq_len, 4

    labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1]*tf.shape(labels)[2],))
    labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1]*tf.shape(labels_mask)[2],))
    predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1]*tf.shape(predictions)[2],))

    # labels_mask.shape: batch, seq_len
    indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
    labels_ = tf.gather(labels, indices)
    predictions_ = tf.gather(predictions, indices)

    # loss_.shape: batch*seq_len*3
    loss_ = mse_loss_func(labels_, predictions_)

    return loss_

def cal_accuracy(name, predictions, labels, labels_mask, total_len):

    if name == "SS8":

        labels = labels[:,:,:8]
        labels_mask = labels_mask[:,:,:8]


        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)
        assert total_len == labels_.shape[0] == predictions_.shape[0]

        accuracy = tf.cast(tf.equal(tf.argmax(labels_,1), tf.argmax(predictions_,1)), tf.float32)

        return accuracy.numpy()

    elif name == "SS3":

        labels = labels[:,:,30:33]
        labels_mask = labels_mask[:,:,30:33]

        #shape labels and labels_mask (batch, seq len, 3)
        #batch : 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        assert total_len == labels_.shape[0] == predictions_.shape[0]

        accuracy = tf.cast(tf.equal(tf.argmax(labels_,1), tf.argmax(predictions_,1)), tf.float32)

        return accuracy.numpy()

    elif name == "PhiPsi":

        labels = labels[:,:,24:26]
        labels_mask = labels_mask[:,:,24:26]

        # labels.shape: batch, seq_len, 2
        # predictions.shape: batch, seq_len, 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], 2))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], 2))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], 4))

        # labels.shape: batch*seq_len, 2
        # predictions.shape: batch*seq_len, 4
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions = tf.gather(predictions, indices)
        assert total_len == labels_.shape[0] == predictions.shape[0]

        labels_ = labels_.numpy()
        predictions = predictions.numpy()

        # predictions.shape: batch*seq_len, 2
        predictions_ = np.zeros((np.shape(predictions)[0], 2))
        predictions_[:,0] = np.rad2deg(
            np.arctan2(predictions[:,0], predictions[:,1]))
        predictions_[:,1] = np.rad2deg(
            np.arctan2(predictions[:,2], predictions[:,3]))

        phi_diff = labels_[:,0] - predictions_[:,0]
        phi_diff[np.where(phi_diff<-180)] += 360
        phi_diff[np.where(phi_diff>180)] -= 360
        mae_phi = np.abs(phi_diff)

        psi_diff = labels_[:,1] - predictions_[:,1]
        psi_diff[np.where(psi_diff<-180)] += 360
        psi_diff[np.where(psi_diff>180)] -= 360
        mae_psi = np.abs(psi_diff)

        return mae_phi, mae_psi

    elif name == "ASA":

        #size should be: (batch=4, seq_len, 1)
        labels = tf.expand_dims(labels[:,:,23],-1)
        labels_mask = tf.expand_dims(labels_mask[:,:,23],-1)

        #reshape to: batch*seq_len
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1]*tf.shape(labels)[2],))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1]*tf.shape(labels_mask)[2],))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1]*tf.shape(predictions)[2],))

        #give it dimension batch*seq_len, 1
        labels = tf.expand_dims(labels,-1)
        labels_mask = tf.expand_dims(labels_mask,-1)
        predictions = tf.expand_dims(predictions,-1)

        #give labels shape (batch*seq_len,)
        labels_mask = labels_mask[:,0]

        #size indices (number zeros in labels_mask,)
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        #size labels_ and predictions_: (number zeros in labels_mask,1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        accuracy = tf.cast(tf.equal(tf.argmax(labels_,1), tf.argmax(predictions_,1)), tf.float32)

        labels_pcc =  tf.reshape(labels_, [tf.shape(labels_)[0]*tf.shape(labels_)[1]])
        predictions_pcc = tf.reshape(predictions_, [tf.shape(predictions_)[0]*tf.shape(predictions_)[1]])
        pearson_np = np.corrcoef(predictions_pcc, labels_pcc)[0][1]

        return pearson_np

    elif name == "Buried":
        labels = labels[:,:,33:35]
        labels_mask = labels_mask[:,:,33:35]

        #shape labels and labels_mask (batch, seq len, 3)
        #batch : 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2])))

        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        assert total_len == labels_.shape[0] == predictions_.shape[0]

        accuracy = tf.cast(tf.equal(tf.argmax(labels_,1), tf.argmax(predictions_,1)), tf.float32)

        return accuracy.numpy()

    elif name == "PPI":
        labels = labels[:,:,35]
        labels_mask = labels_mask[:,:,35]

        labels = tf.expand_dims(labels, -1)
        labels_mask = tf.expand_dims(labels_mask, -1)

        #batch : 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)

        is_empty = tf.equal(tf.size(indices), 0)

        if is_empty == False:

            labels_ = tf.gather(labels, indices)
            predictions_ = tf.gather(predictions, indices)

            predictions_IF = predictions_[:,1]

            assert labels_.shape[0] == predictions_.shape[0]

            weights = tf.constant([0])

            labels_reshape =tf.reshape(labels_, [tf.shape(labels_)[0]])
            labels_reshape = tf.dtypes.cast(labels_reshape, tf.float32)
            predictions_class = tf.argmax(predictions_,1)
            predictions_class = tf.dtypes.cast(predictions_class, tf.float32)

            accuracy = tf.cast(tf.equal(labels_reshape, predictions_class), tf.float32)

            return accuracy.numpy(), predictions_, labels_, weights

        else:
            return None, None, None, None


def clean_inputs(x, x_mask, dim_input):
    # set 0
    # x.shape: batch, seq_len, dim_input
    # x_mask.shape: batch, seq_len
    x_mask = tf.tile(x_mask[:,:,tf.newaxis], [1, 1, dim_input])
    x_clean = tf.where(tf.math.equal(x_mask, 0), x, x_mask-1)
    return x_clean

def get_output(name, predictions, x_mask, total_len):

    if name == "SS":

        ss_outputs = []

        ss_prediction = tf.nn.softmax(predictions[0])
        for i in predictions[1:]:
            ss_prediction += tf.nn.softmax(i)
        ss_prediction = tf.nn.softmax(ss_prediction)

        x_mask = x_mask.numpy()
        ss_prediction = ss_prediction.numpy()

        max_length = x_mask.shape[1]
        for i in range(x_mask.shape[0]):
            indiv_length = int(max_length-np.sum(x_mask[i]))
            ss_outputs.append(ss_prediction[i][:indiv_length])

        ss_outputs_concat = np.concatenate(ss_outputs, 0)
        assert ss_outputs_concat.shape[0] == total_len

        return ss_outputs, ss_outputs_concat

    elif name == "PhiPsi":

        phi_predictions = []
        psi_predictions = []
        phi_outputs = []
        psi_outputs = []
        for i in predictions:

            # i.shape: batch, seq_len, 4
            i = i.numpy()

            phi_prediction = np.zeros((i.shape[0], i.shape[1], 1))
            psi_prediction = np.zeros((i.shape[0], i.shape[1], 1))

            phi_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,0], i[:,:,1]))
            psi_prediction[:,:,0] = np.rad2deg(np.arctan2(i[:,:,2], i[:,:,3]))

            phi_predictions.append(phi_prediction)
            psi_predictions.append(psi_prediction)

        phi_predictions = np.concatenate(phi_predictions, -1)
        phi_predictions = np.median(phi_predictions, -1)

        psi_predictions = np.concatenate(psi_predictions, -1)
        psi_predictions = np.median(psi_predictions, -1)

        x_mask = x_mask.numpy()
        max_length = x_mask.shape[1]
        for i in range(x_mask.shape[0]):
            indiv_length = int(max_length-np.sum(x_mask[i]))
            phi_outputs.append(phi_predictions[i][:indiv_length])
            psi_outputs.append(psi_predictions[i][:indiv_length])

        phi_outputs_concat = np.concatenate(phi_outputs, 0)
        psi_outputs_concat = np.concatenate(psi_outputs, 0)
        assert phi_outputs_concat.shape[0] == psi_outputs_concat.shape[0] == total_len

        return phi_outputs, psi_outputs, [phi_outputs_concat, psi_outputs_concat]

def error_analyse(name, predictions, labels, labels_mask):

    if name == "Buried":
        labels = labels[:,:,33:35]
        labels_mask = labels_mask[:,:,33:35]

        #shape labels and labels_mask (batch, seq len, 3)
        #batch : 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        #first column buried, second non-buried
        labels_ =  tf.argmax(labels_,1)
        predictions_ = tf.argmax(predictions_,1)

        return labels_, predictions_


    elif name == "ASA":
        #size should be: (batch=4, seq_len, 1)
        labels = tf.expand_dims(labels[:,:,23],-1)
        labels_mask = tf.expand_dims(labels_mask[:,:,23],-1)

        #reshape to: batch*seq_len
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1]*tf.shape(labels)[2],))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1]*tf.shape(labels_mask)[2],))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1]*tf.shape(predictions)[2],))

        #give it dimension batch*seq_len, 1
        labels = tf.expand_dims(labels,-1)
        labels_mask = tf.expand_dims(labels_mask,-1)
        predictions = tf.expand_dims(predictions,-1)

        #give labels shape (batch*seq_len,)
        labels_mask = labels_mask[:,0]

        #size indices (number zeros in labels_mask,)
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        #size labels_ and predictions_: (number zeros in labels_mask,1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        labels_pcc =  tf.reshape(labels_, [tf.shape(labels_)[0]*tf.shape(labels_)[1]])
        predictions_pcc = tf.reshape(predictions_, [tf.shape(predictions_)[0]*tf.shape(predictions_)[1]])
        # pearson_np = np.corrcoef(predictions_pcc, labels_pcc)[0][1]

        return labels_pcc, predictions_pcc

    elif name == "SS8":

        labels = labels[:,:,:8]
        labels_mask = labels_mask[:,:,:8]


        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        labels_ =  tf.argmax(labels_,1)
        predictions_ = tf.argmax(predictions_,1)

        return labels_, predictions_

    elif name == "SS3":

        labels = labels[:,:,30:33]
        labels_mask = labels_mask[:,:,30:33]

        #shape labels and labels_mask (batch, seq len, 3)
        #batch : 4
        labels = tf.reshape(labels, (tf.shape(labels)[0]*tf.shape(labels)[1], tf.shape(labels)[2]))
        labels_mask = tf.reshape(labels_mask, (tf.shape(labels_mask)[0]*tf.shape(labels_mask)[1], tf.shape(labels_mask)[2]))
        predictions = tf.reshape(predictions, (tf.shape(predictions)[0]*tf.shape(predictions)[1], tf.shape(predictions)[2]))

        # labels_mask.shape: batch, seq_len
        labels_mask = labels_mask[:,0]
        indices = tf.squeeze(tf.where(tf.math.equal(labels_mask, 0)), 1)
        labels_ = tf.gather(labels, indices)
        predictions_ = tf.gather(predictions, indices)

        labels_ =  tf.argmax(labels_,1)
        predictions_ = tf.argmax(predictions_,1)

        return labels_, predictions_
