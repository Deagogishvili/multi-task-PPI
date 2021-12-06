# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang
@edit: Henriette Capel 02 Mar 2021


"""
import time
from my_model_mylabels_prob import Model
import tensorflow as tf
import numpy as np
from tensorflow import keras
from utils_mylabels_prob import InputReader, cal_accuracy, error_analyse

import click
import random
import os
import csv

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score


################################################################################
@click.command()
@click.option('--tensorboard-dir', default=None)
@click.option('--training_path', default = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/training_ppi_pdb.txt")
@click.option('--validation_path', default = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/validation_ppi_pdb.txt")
@click.option('--test_path', default = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/test_ppi_pdb.txt")
@click.option('--output_erroranalyse', default = None)
@click.option('--lr', default = 1e-4)
@click.option('--class-imbalance-major', default = 1.0)
@click.option('--class-imbalance-minor', default = 1.0)

################################################################################

def main(tensorboard_dir, training_path, validation_path, test_path, output_erroranalyse, lr, class_imbalance_major, class_imbalance_minor):
    print("tensorboard directory is: {}".format(tensorboard_dir))
    print("Path to list of training samples: {}".format(training_path))
    print("Path to list of validation samples: {}".format(validation_path))
    print("learning rate: {}".format(lr))
    print("class imbalance: {}".format([class_imbalance_major, class_imbalance_minor]))

    #train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_precision = tf.keras.metrics.Precision(name='train_precision')
    train_recall = tf.keras.metrics.Recall(name='train_recall')
    train_FP = tf.keras.metrics.FalsePositives(name='train_FP')
    train_TP = tf.keras.metrics.TruePositives(name = 'train_TP')
    train_FN = tf.keras.metrics.FalseNegatives(name = 'train_FN')
    train_TN = tf.keras.metrics.TrueNegatives(name = 'train_TN')

    #val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    val_precision = tf.keras.metrics.Precision(name='val_precision')
    val_recall = tf.keras.metrics.Recall(name='val_recall')
    val_FP = tf.keras.metrics.FalsePositives(name='val_FP')
    val_TP = tf.keras.metrics.TruePositives(name = 'val_TP')
    val_FN = tf.keras.metrics.FalseNegatives(name = 'val_FN')
    val_TN = tf.keras.metrics.TrueNegatives(name = 'val_TN')

    test_precision = tf.keras.metrics.Precision(name='test_precision')
    test_recall = tf.keras.metrics.Recall(name='test_recall')
    test_FP = tf.keras.metrics.FalsePositives(name='test_FP')
    test_TP = tf.keras.metrics.TruePositives(name = 'test_TP')
    test_FN = tf.keras.metrics.FalseNegatives(name = 'test_FN')
    test_TN = tf.keras.metrics.TrueNegatives(name = 'test_TN')

    #parameters of training
    batch_size = 4
    epochs = 40
    early_stop = 4
    input_normalization = True
    learning_rate = lr

    params = {}
    params["d_input"] = 76
    params["d_ss8_output"] = 8
    params["d_ss3_output"] = 3
    params["d_phipsi_output"] = 4
    params["d_csf_output"] = 3
    params["d_asa_output"] = 1
    params["d_rota_output"] = 8
    params["d_buried_output"] = 2
    params["d_ppi_output"] = 2
    params["dropout_rate"] = 0.25

    #parameters of transfomer model
    params["transfomer_layers"] = 2
    params["transfomer_num_heads"] = 4

    #parameters of birnn model
    params["lstm_layers"] = 4
    params["lstm_units"] = 1024

    #parameters of cnn model
    params["cnn_layers"] = 5
    params["cnn_channels"] = 32

    #params["save_path"] = r'./models'
    params["save_path"] = "/var/scratch/hcl700/Major_Internship/multi_task/models"

    #params["weight_loss_phipsi"] = 4
    params["weight_loss_phipsi"] = 1
    #params["weight_loss_csf"] = 0.1
    params["weight_loss_csf"] = 1
    #params["weigth_loss_asa"] = 3
    params["weigth_loss_asa"] = 1
    params["weight_loss_ss8"] = 1
    params["weight_loss_ss3"] = 1
    params["weight_loss_rota"] = 1
    params["weight_loss_buried"] = 1
    params["weight_loss_ppi"] = 1

    #first class weight for majority class, second for minority class
    params['class_weights'] = [class_imbalance_major, class_imbalance_minor]

    ############################################################################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), len(logical_gpus))

    ############################################################################
    #Use my labels
    # train_list_path = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/training_pdb.txt"
    # val_list_path = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/validation_pdb.txt"
    # test_list_path = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/test_pdb.txt"

    train_list_path = training_path
    val_list_path = validation_path
    test_list_path = test_path

    fastas_files_path_trainval = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/trainval_fastas"
    inputs_files_path_trainval = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/trainval_inputs"
    labels_files_path_trainval = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/trainval_my_labels"

    fastas_files_path_test = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/test_fastas"
    inputs_files_path_test = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/test_inputs"
    labels_files_path_test = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/test_my_labels"

    """
    labels shape:
    ss_labels = labels[:,:8]
    csf_labels = labels[:,8:11]
    phipsi_labels = labels[:,11:15]
    dihedrals_labels = labels[:,15:23]
    asa_labels = labels[:,23]             (/100)
    real_phipsidihedrals=labels[:,24:30]
    ss3 = labels[:,30:33]
    buried = labels[:,33]
    nonburied = labels[:,34]
    ppi = labels[:,35]
    """

    ############################################################################
    model_IFBU = Model(params=params, name="IFBU")

    train_reader = InputReader(data_list=train_list_path,
                               inputs_files_path=inputs_files_path_trainval,
                               labels_files_path=labels_files_path_trainval,
                               fastas_files_path=fastas_files_path_trainval,
                               num_batch_size=batch_size,
                               input_norm=input_normalization,
                               shuffle=True,
                               data_enhance=True)

    val_reader = InputReader(data_list=val_list_path,
                             inputs_files_path=inputs_files_path_trainval,
                             labels_files_path=labels_files_path_trainval,
                             fastas_files_path=fastas_files_path_trainval,
                             num_batch_size=batch_size,
                             input_norm=input_normalization,
                             shuffle=False,
                             data_enhance=False)

    test_reader = InputReader(data_list=test_list_path,
                              inputs_files_path=inputs_files_path_test,
                              labels_files_path=labels_files_path_test,
                              fastas_files_path=fastas_files_path_test,
                              num_batch_size=batch_size,
                              input_norm=input_normalization,
                              shuffle=False,
                              data_enhance=False)

    lr = tf.Variable(tf.constant(learning_rate), name='lr', trainable=False)
    optimizer = keras.optimizers.Adam(lr=lr)

    #Used by tensorboard to write loss values to
    if tensorboard_dir:
        print("Creating Tensorboard")
        #tensorboard_dir_worker = tensorboard_dir + '/' + str(strategy.cluster_resolver.task_id)
        writer = tf.summary.create_file_writer(tensorboard_dir + "/" + "train")
        val_writer = tf.summary.create_file_writer(tensorboard_dir + "/" + "val")
        test_writer = tf.summary.create_file_writer(tensorboard_dir + "/" + "test")
    else:
        writer = None


    def train_step(x, x_mask, y, y_mask):

        ppi_predictions = buried_predictions = None

        with tf.GradientTape() as tape:
             ppi_predictions, buried_predictions, loss = model_IFBU.inference(x, x_mask, y, y_mask, training=True)

        trainable_variables = model_IFBU.transformer.trainable_variables + \
            model_IFBU.cnn.trainable_variables + model_IFBU.birnn.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, trainable_variables))

        return loss, ppi_predictions, buried_predictions

    def infer_step(x, x_mask):

        ppi_predictions = buried_predictions = None

        ppi_predictions, buried_predictions, _ = model_IFBU.inference(x, x_mask, y, y_mask, training=False)

        return ppi_predictions, buried_predictions

    def correct_formatting(targets, predictions):
        #No amino acids already removed
        pred_class = tf.argmax(predictions, 1)
        pred_prob = tf.reshape(predictions[:,1], [tf.shape(predictions)[0]])
        tar = tf.reshape(targets, [tf.shape(targets)[0]])

        pred_class = pred_class.numpy()
        pred_prob = pred_prob.numpy()
        tar = tar.numpy()

        return tar, pred_class, pred_prob

    def metrices(targets, predictions, predictions_probabilities_interface, set):
        fpr , tpr , thresholds = roc_curve(targets, predictions_probabilities_interface)
        auc_roc = roc_auc_score(targets,predictions_probabilities_interface)
        precision = precision_score(targets, predictions, zero_division = 0)
        recall = recall_score(targets, predictions, zero_division = 0)
        precision_list , recall_list , thresholds_PR = precision_recall_curve(targets, predictions_probabilities_interface)
        auc_pr = auc(recall_list, precision_list)

        P = sum(targets)
        N = len(targets)-P
        fraction_positive = P / (P+N)

        if set == "train":
            train_precision.update_state(targets, predictions)
            train_recall.update_state(targets, predictions)
            train_TP.update_state(targets, predictions)
            train_FP.update_state(targets, predictions)
            train_FN.update_state(targets, predictions)
            train_TN.update_state(targets, predictions)
        elif set == "val":
            val_precision.update_state(targets, predictions)
            val_recall.update_state(targets, predictions)
            val_TP.update_state(targets, predictions)
            val_FP.update_state(targets, predictions)
            val_FN.update_state(targets, predictions)
            val_TN.update_state(targets, predictions)
        elif set == "test":
            test_precision.update_state(targets, predictions)
            test_recall.update_state(targets, predictions)
            test_TP.update_state(targets, predictions)
            test_FP.update_state(targets, predictions)
            test_FN.update_state(targets, predictions)
            test_TN.update_state(targets, predictions)

        return fpr, tpr, auc_roc, precision, recall, precision_list, recall_list, fraction_positive, auc_pr

    def plotting_roc(fpr, tpr, auc_score, path, epoch):
        lw = 1
        plt.figure(1)           #write all ROC to this output file
        plt.plot(fpr, tpr, label= 'ROC epoch {} (area = {:.3f})'.format(epoch + 1, auc_score))
        plt.plot([0,1], [0,1], color="navy", lw=lw, linestyle='--')
        plt.ylabel('True positive rate', size = 10)
        plt.xlabel("False positive rate", size = 10)
        plt.legend(loc="lower right", fontsize= 8)
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.savefig(path)

    def plotting_pr(recall, precision, fraction_positive, path, epoch):
        lw = 1
        plt.figure(2)       ##write all PR plots to same file
        plt.plot(recall, precision, label = 'PR epoch {}'.format(epoch + 1))
        plt.hlines(fraction_positive, 0, 1, color="navy", lw=lw, linestyle='--')
        plt.ylabel('precision', size = 10)
        plt.xlabel("recall", size = 10)
        plt.legend(loc="upper right", fontsize= 8)
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.savefig(path)

    def write_output_plots(auc_roc, auc_pr, fraction_positive, targets, predictions, pred_prob_IF, path):
        file = open(path, "w")
        file.write("auc_roc: " + str(auc_roc) + "\n")
        file.write("auc_pr: " + str(auc_pr)+ "\n")
        file.write("fraction_positive: " + str(fraction_positive) + "\n")
        file.write("targets: " + str(targets) + "\n")
        file.write("predictions: " + str(predictions) + "\n")
        file.write("pred_prob_IF: " + str(pred_prob_IF) + "\n")
        file.close()

    def error_analyse_store(info_tflist, save_list, index_list):
        info_list = info_tflist.tolist()

        for i in range(0, len(index_list)-1):
            info_seq = info_list[index_list[i]:index_list[i+1]]
            save_list.append(info_seq)
        return save_list

    best_acc = 0
    best_auc_roc = 0
    step_train = 0
    for epoch in range(epochs):

        #======================Train======================
        accuracy_train_ppi = []
        accuracy_train_buried = []

        #train_accuracy.reset_states()
        train_precision.reset_states()
        train_recall.reset_states()
        train_TP.reset_states()
        train_FP.reset_states()
        train_FN.reset_states()
        train_TN.reset_states()

        target_list_train = []
        prediction_list_train = []
        probability_list_train = []

        for step, filenames_batch in enumerate(train_reader.dataset):
            step_train += 1

            start_time = time.time()
            # x (batch, max_len, 76)
            # x_mask (batch, max_len)
            # encoder_padding_mask (batch, 1, 1, max_len)
            # y (batch, max_len, 30)
            # y_mask (batch, max_len, 30)
            filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
                train_reader.read_file_from_disk(filenames_batch)


            #check if inputs_total_len equals labels_total_len otherwise AssertionError
            assert inputs_total_len == labels_total_len

            loss, ppi_predictions, buried_predictions = train_step(x, x_mask, y, y_mask)

            #ppi prediction
            acc_ppi, pred, tar, weights = cal_accuracy("PPI", ppi_predictions, y, y_mask, total_len = inputs_total_len)
            if acc_ppi is not None:
                accuracy_train_ppi.extend(acc_ppi)
                target_list, prediction_list, probability_list = correct_formatting(tar, pred)
                target_list_train.extend(target_list)
                prediction_list_train.extend(prediction_list)
                probability_list_train.extend(probability_list)

            accuracy_train_buried.extend(
                cal_accuracy("Buried", buried_predictions, y, y_mask, total_len = inputs_total_len))


            run_time = time.time() - start_time

            if step % 20  == 0:
                #train_accuracy(tar, pred)

                if len(target_list_train) > 0:
                    if sum(target_list_train) >= 1:
                        fpr, tpr, auc_roc, precision, recall, precision_list, recall_list, fraction_positive, auc_pr = metrices(target_list_train, prediction_list_train, probability_list_train, "train")

                        print('Epoch: %d, step: %d, loss: %3.3f, acc_bur: %3.4f, acc_ppi: %3.4f, AUC_roc: %3.4f, AUC_pr: %3.4f, prec: %3.4f, recall: %3.4f, TP: %0.1f, FP: %0.1f, TN: %0.1f, FN: %0.1f, time: %3.3f'
                            % (epoch, step, loss, np.mean(accuracy_train_buried), np.mean(accuracy_train_ppi), auc_roc, auc_pr, train_precision.result(), train_recall.result(), train_TP.result(), train_FP.result(), train_TN.result(), train_FN.result(), run_time))
                        if writer:
                            with writer.as_default():
                                tf.summary.scalar('loss', loss, step_train)
                                tf.summary.scalar('accuracy_ppi', np.mean(accuracy_train_ppi), step_train)
                                tf.summary.scalar("auc_roc", auc_roc, step_train)
                                tf.summary.scalar("auc_pr", auc_pr, step_train)
                                tf.summary.scalar("accuracy_buried", np.mean(accuracy_train_buried), step_train)
                    else:
                        print("ERROR: target list contains no interface positions")
                        print(len(target_list_train))

                else:
                    print("None ppi seen")
                    print(len(target_list_train))
        #======================Val======================
        accuracy_val_ppi = []
        accuracy_val_buried = []

        #val_accuracy.reset_states()
        val_precision.reset_states()
        val_recall.reset_states()
        val_TP.reset_states()
        val_FP.reset_states()
        val_FN.reset_states()
        val_TN.reset_states()

        target_list_val = []
        prediction_list_val = []
        probability_list_val = []

        #filenames_list_val = []

        start_time = time.time()
        for step, filenames_batch in enumerate(val_reader.dataset):

            filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
                val_reader.read_file_from_disk(filenames_batch)
            #filenames same as filenames_batch in validation

            assert inputs_total_len == labels_total_len

            ppi_predictions, buried_predictions = infer_step(x, x_mask)

            #ppi predictions
            acc_ppi, pred, tar, weights = cal_accuracy("PPI", ppi_predictions, y, y_mask, total_len = inputs_total_len)
            if acc_ppi is not None:
                accuracy_val_ppi.extend(acc_ppi)
                target_list, prediction_list, probability_list = correct_formatting(tar, pred)
                #Use extend not append to add all elements to the big list.
                target_list_val.extend(target_list)
                prediction_list_val.extend(prediction_list)
                probability_list_val.extend(probability_list)


            accuracy_val_buried.extend(
                cal_accuracy("Buried", buried_predictions, y, y_mask, total_len = inputs_total_len))


        #End of validation
        run_time = time.time() - start_time
        fpr_val, tpr_val, auc_roc_val, precision_val, recall_val, precision_list_val, recall_list_val, fraction_positive_val, auc_pr_val = metrices(target_list_val, prediction_list_val, probability_list_val, "val")

        print('Epoch: %d, lr: %s, acc_bur: %3.4f, acc_ppi: %3.4f, AUC_ROC: %3.4f, AUC_PR: %3.4f, prec: %3.4f, recall: %3.4f, TP: %0.1f, FP: %0.1f, TN: %0.1f, FN: %0.1f, time: %3.3f'
            % (epoch, str(lr.numpy()), np.mean(accuracy_val_buried), np.mean(accuracy_val_ppi), auc_roc_val, auc_pr_val, val_precision.result(), val_recall.result(), val_TP.result(), val_FP.result(), val_TN.result(), val_FN.result(), run_time))

        #optimise for AUC score
        #if  np.mean(accuracy_val_ppi) > best_acc:
        if auc_roc_val > best_auc_roc:
            best_acc =  np.mean(accuracy_val_ppi)
            best_epoch = epoch
            best_auc_roc = auc_roc_val
            best_auc_pr = auc_pr_val

            best_precision = precision_val
            best_recall = recall_val
            best_TP = val_TP.result()
            best_FP = val_FP.result()
            best_TN = val_TN.result()
            best_FN = val_FN.result()
            best_target_list = target_list_val
            best_prediction_list = prediction_list_val
            best_probabilities_list = probability_list_val

            best_acc_bur = np.mean(accuracy_val_buried)

            best_fpr = fpr_val
            best_tpr = tpr_val
            best_recall_list = recall_list_val
            best_precision_list = precision_list_val
            best_fraction_positive = fraction_positive_val

            model_IFBU.save_model()

        else:
            lr.assign(lr/2)
            early_stop -= 1

        if early_stop == 0:
            break

        # END OF EPOCH SUMMARY
        if writer:
            with writer.as_default():
                tf.summary.scalar('epoch_loss', loss, epoch)
                tf.summary.scalar('epoch_accuracy_ppi', np.mean(accuracy_train_ppi), epoch)
                tf.summary.scalar("epoch_auc_roc", auc_roc, epoch)
                tf.summary.scalar("epoch_auc_pr", auc_pr, epoch)
                tf.summary.scalar("epoch_accuracy_bur", np.mean(accuracy_train_buried), epoch)

        if val_writer:
            with val_writer.as_default():
                tf.summary.scalar('learning_rate', lr.numpy(), epoch)
                tf.summary.scalar('epoch_accuracy_ppi', np.nanmean(accuracy_val_ppi), epoch)
                tf.summary.scalar("epoch_auc_roc", auc_roc_val, epoch)
                tf.summary.scalar("epoch_auc_pr", auc_pr_val, epoch)
                tf.summary.scalar('val_precision_ppi', val_precision.result(), epoch)
                tf.summary.scalar('val_recall_ppi', val_recall.result(), epoch)
                tf.summary.scalar('val_TP_ppi', val_TP.result(), epoch)
                tf.summary.scalar('val_FP_ppi', val_FP.result(), epoch)
                tf.summary.scalar('val_TN_ppi', val_TN.result(), epoch)
                tf.summary.scalar('val_FN_ppi', val_FN.result(), epoch)
                tf.summary.scalar("epoch_accuracy_bur", np.mean(accuracy_val_buried), epoch)


    #print("best_val_ppi_accuracy:", best_acc)
    print("best auc: ", best_auc_roc)
    #Print the best epoch
    print("Corresponding best performance measures:")
    print('Epoch: %d, acc_bur: %3.4f, acc_ppi: %3.4f, AUC_ROC: %3.4f, AUC_PR: %3.4f, prec: %3.4f, recall: %3.4f, TP: %0.1f, FP: %0.1f, TN: %0.1f, FN: %0.1f'
          % (best_epoch, best_acc_bur, best_acc, best_auc_roc, best_auc_pr, best_precision, best_recall, best_TP, best_FP, best_TN, best_FN))

    #Print the curves for that epoch
    # path_fig_roc = tensorboard_dir + "/ROC.png"
    # plotting_roc(best_fpr, best_tpr, best_auc, path_fig_roc, best_epoch)
    # path_fig_pr = tensorboard_dir + "/PR.png"
    # plotting_pr(best_recall_list, best_precision_list, best_fraction_positive, path_fig_pr, best_epoch)
    path_output_write = tensorboard_dir + "/output_" + str(epoch+1) + ".txt"
    write_output_plots(best_auc_roc, best_auc_pr, best_fraction_positive, best_target_list, best_prediction_list, best_probabilities_list, path_output_write)

    #======================Test======================

    model_IFBU_test = Model(params=params, name="IFBU")
    model_IFBU_test.load_model()

    def test_infer_step(x, x_mask):

        ppi_predictions = buried_predictions = None

        ppi_predictions, buried_predictions, _ = model_IFBU_test.inference(x, x_mask, y, y_mask, training=False)

        return ppi_predictions, buried_predictions

    accuracy_test_ppi = []
    accuracy_test_buried = []
    target_list_test = []
    prediction_list_test = []
    probability_list_test = []

    #Needed for error analyse
    filenames_list = []
    target_ppi_list_error = []
    prediction_ppi_list_error = []
    probability_ppi_list_error = []
    target_bur_list_error = []
    prediction_bur_list_error = []
    sequences_total = []
    length_seq_total = []

    acc_ppi_error_list = []
    auc_roc_error_list = []
    auc_pr_error_list = []
    precision_error_list = []
    recall_error_list = []
    acc_bur_error_list = []

    start_time = time.time()
    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
            test_reader.read_file_from_disk(filenames_batch)

        ######
        filenames_list.extend(filenames)
        length_seq_batch = []
        for filename in filenames:
            fasta_ = open(os.path.join(fastas_files_path_test, filename + ".fasta"), "r")
            for line in fasta_:
                if line[0] == ">":
                    length = line.strip("\n").split(" ")[1]
                    length_seq_batch.append(int(length))
                else:
                    seq = line.strip("\n")
                    aa_seq = []
                    aa_seq[:0] = seq
                    sequences_total.append(aa_seq)
            fasta_.close()
        length_seq_total.extend(length_seq_batch)
        ######

        assert inputs_total_len == labels_total_len

        ppi_predictions, buried_predictions = test_infer_step(x, x_mask)

        acc_ppi, pred, tar, weights = cal_accuracy("PPI", ppi_predictions, y, y_mask, total_len = inputs_total_len)
        if acc_ppi is not None:
            accuracy_test_ppi.extend(acc_ppi)
            target_list_ppi, prediction_list_ppi, probability_list_ppi = correct_formatting(tar, pred)

            target_list_test.extend(target_list_ppi)
            prediction_list_test.extend(prediction_list_ppi)
            probability_list_test.extend(probability_list_ppi)

            ##### Error analyse ##########
            tar_bur, pred_bur = error_analyse("Buried", buried_predictions, y, y_mask)
            tar_bur = tar_bur.numpy()
            pred_bur = pred_bur.numpy()

            index_list = [0]
            for i in range(0,len(length_seq_batch)):
                value = index_list[i] + length_seq_batch[i]
                index_list.append(value)

            target_ppi_list_error = error_analyse_store(target_list_ppi, target_ppi_list_error, index_list)
            prediction_ppi_list_error = error_analyse_store(prediction_list_ppi, prediction_ppi_list_error, index_list)
            probability_ppi_list_error = error_analyse_store(probability_list_ppi, probability_ppi_list_error, index_list)

            target_bur_list_error = error_analyse_store(tar_bur, target_bur_list_error, index_list)
            prediction_bur_list_error = error_analyse_store(pred_bur, prediction_bur_list_error, index_list)

        accuracy_test_buried.extend(
            cal_accuracy("Buried", buried_predictions, y, y_mask, total_len = inputs_total_len))

    run_time = time.time() - start_time
    fpr_test, tpr_test, auc_roc_test, precision_test, recall_test, precision_list_test, recall_list_test, fraction_positive_test, auc_pr_test = metrices(target_list_test, prediction_list_test, probability_list_test, "test")

    print("Performance on the testset")
    print('acc_bur: %3.4f, acc_ppi: %3.4f, AUC_roc: %3.4f, AUC_pr %3.4f, prec: %3.4f, recall: %3.4f, TP: %0.1f, FP: %0.1f, TN: %0.1f, FN: %0.1f, time: %3.3f'
        % (np.mean(accuracy_test_buried), np.mean(accuracy_test_ppi), auc_roc_test, auc_pr_test, test_precision.result(), test_recall.result(), test_TP.result(), test_FP.result(), test_TN.result(), test_FN.result(), run_time))


    # #### ERROR analyse ####
    #
    # if output_erroranalyse:
    #     print("Perform error analyse")
    #     for i in range(0,len(filenames_list)):
    #         fpr, tpr, auc_roc, precision, recall, precision_list, recall_list, fraction_positive, auc_pr = metrices(target_ppi_list_error[i], prediction_ppi_list_error[i], probability_ppi_list_error[i], "None")
    #         auc_roc_error_list.append(auc_roc)
    #         auc_pr_error_list.append(auc_pr)
    #         precision_error_list.append(precision)
    #         recall_error_list.append(recall)
    #
    #         acc_ppi = accuracy_score(target_ppi_list_error[i], prediction_ppi_list_error[i])
    #         acc_ppi_error_list.append(acc_ppi)
    #
    #         acc_bur = accuracy_score(target_bur_list_error[i], prediction_bur_list_error[i])
    #         acc_bur_error_list.append(acc_bur)
    #
    #         output_path = output_erroranalyse
    #         os.makedirs(output_path, exist_ok=True)
    #
    #         sumfile_path = output_path + "/IFBU_summary.csv"
    #         with open(sumfile_path, "w") as f:
    #             writer_sum = csv.writer(f, delimiter='\t')
    #             writer_sum.writerow(("ID", "acc_ppi", "auc_roc_ppi", "auc_pr_ppi", "precision_ppi", "recall_ppi", "acc_bur"))
    #             writer_sum.writerows(zip(filenames_list, acc_ppi_error_list, auc_roc_error_list, auc_pr_error_list, precision_error_list, recall_error_list, acc_bur_error_list))
    #
    #
    #     for i in range(0, len(filenames_list)):
    #         list_filename = [filenames_list[i]]*length_seq_total[i]
    #         list_aa_seq = sequences_total[i]
    #         list_tar = map(int, target_ppi_list_error[i])
    #         list_pred = prediction_ppi_list_error[i]
    #         list_prob = probability_ppi_list_error[i]
    #
    #         datafile_path = output_path + "/" + filenames_list[i] + ".csv"
    #         with open(datafile_path, "w") as f:
    #             writer_out = csv.writer(f, delimiter='\t')
    #             writer_out.writerow(("ID", "AA", "PPI_tar", "PPI_pred", "PPI_prob"))
    #             writer_out.writerows(zip(list_filename, list_aa_seq, list_tar, list_pred, list_prob))

if __name__ == '__main__':
    main()
