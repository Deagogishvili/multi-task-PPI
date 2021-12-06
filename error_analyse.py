# -*- coding: utf-8 -*-
"""
Created on 23 april 2021
Script to perform error analyse on saved model

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
@click.option('--model_name')
@click.option('--output_path', default = None)
################################################################################
def main(model_name, output_path):
    print("Name model: {}".format(model_name))
    print("Output error analyse path: {}".format(output_path))

    #parameters
    batch_size = 4
    input_normalization = True

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

    ############################################################################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), len(logical_gpus))

    ############################################################################

    test_ppi_list_path = "/var/scratch/hcl700/Major_Internship/Data/My_database/Data/test_ppi_pdb.txt"

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

    erroranalyse_reader = InputReader(data_list=test_list_path,
                              inputs_files_path=inputs_files_path_test,
                              labels_files_path=labels_files_path_test,
                              fastas_files_path=fastas_files_path_test,
                              num_batch_size=batch_size,
                              input_norm=input_normalization,
                              shuffle=False,
                              data_enhance=False)

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

    def error_analyse_store(info_tflist, save_list, index_list):
        info_list = info_tflist.tolist()

        for i in range(0, len(index_list)-1):
            info_seq = info_list[index_list[i]:index_list[i+1]]
            save_list.append(info_seq)
        return save_list

    #=====================Error analyse================
    #NOTE: need different one because of problems when using dataset including proteins without ppi annotations
    model_IFBUSA_error = Model(params=params, name= model_name)
    model_IFBUSA_error.load_model()

    def test_infer_step(x, x_mask):

        ppi_predictions = asa_predictions = buried_predictions = None

        ppi_predictions, buried_predictions, asa_predictions, _ = \
            model_IFBUSA_error.inference(x, x_mask, y, y_mask, training=False)

        return ppi_predictions, buried_predictions, asa_predictions

    ### ERROR analyse
    filenames_list = []
    target_ppi_list_error = []
    prediction_ppi_list_error = []
    probability_ppi_list_error = []

    target_bur_list_error = []
    prediction_bur_list_error = []
    target_asa_list_error = []
    prediction_asa_list_error = []

    sequences_total = []
    length_seq_total = []

    acc_ppi_error_list = []
    auc_roc_error_list = []
    auc_pr_error_list = []
    precision_error_list = []
    recall_error_list = []
    acc_bur_error_list = []
    pcc_asa_error_list = []

    start_time = time.time()
    for step, filenames_batch in enumerate(erroranalyse_reader.dataset):

        filenames, x, x_mask, y, y_mask, inputs_total_len, labels_total_len = \
            test_reader.read_file_from_disk(filenames_batch)

        assert inputs_total_len == labels_total_len

        ppi_predictions, buried_predictions, asa_predictions = test_infer_step(x, x_mask)

        #ppi predictions
        acc_ppi, pred, tar, weights = cal_accuracy("PPI", ppi_predictions, y, y_mask, total_len = inputs_total_len)
        if acc_ppi is not None:

            #### error: possible that not all proteins in one batch has ppi annotations.
            # werkt dus nog niet voor all data
            #length target ppi is shorter than sum of all lengths ppi in that case. So that is good.
            # How to select onl the once with annotations?

            accuracy_test_ppi.extend(acc_ppi)
            target_list_ppi, prediction_list_ppi, probability_list_ppi = correct_formatting(tar, pred)
            #Use extend not append to add all elements to the big list.
            target_list_test.extend(target_list_ppi)
            prediction_list_test.extend(prediction_list_ppi)
            probability_list_test.extend(probability_list_ppi)

            ######
            filenames_batch_ppi = []
            length_seq_batch = []
            for filename in filenames:
                #only add the sequences that has ppi annotations
                if filename in filenames_ppi_test:
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
            filenames_list.extend(filenames)

            assert sum(length_seq_batch) == len(target_list_ppi)

            ##### Error analyse ##########
            tar_bur, pred_bur = error_analyse("Buried", buried_predictions, y, y_mask)
            tar_bur = tar_bur.numpy()
            pred_bur = pred_bur.numpy()

            tar_asa, pred_asa = error_analyse("ASA", buried_predictions, y, y_mask)
            tar_asa = tar_asa.numpy()
            pred_asa = pred_asa.numpy()

            index_list = [0]
            for i in range(0,4):
                value = index_list[i] + length_seq_batch[i]
                index_list.append(value)

            target_ppi_list_error = error_analyse_store(target_list_ppi, target_ppi_list_error, index_list)
            prediction_ppi_list_error = error_analyse_store(prediction_list_ppi, prediction_ppi_list_error, index_list)
            probability_ppi_list_error = error_analyse_store(probability_list_ppi, probability_ppi_list_error, index_list)

            target_bur_list_error = error_analyse_store(tar_bur, target_bur_list_error, index_list)
            prediction_bur_list_error = error_analyse_store(pred_bur, prediction_bur_list_error, index_list)

            target_asa_list_error = error_analyse_store(tar_asa, target_asa_list_error, index_list)
            prediction_asa_list_error = error_analyse_store(pred_asa, prediction_asa_list_error, index_list)

        ###### for total test performance
        accuracy_test_bur.extend(
            cal_accuracy("Buried", buried_predictions, y, y_mask, total_len = inputs_total_len))

        pearson_asa = cal_accuracy("ASA", asa_predictions, y, y_mask, total_len = inputs_total_len)
        pearson_test_asa.append(pearson_asa)

    run_time = time.time() - start_time
    fpr_test, tpr_test, auc_roc_test, precision_test, recall_test, precision_list_test, recall_list_test, fraction_positive_test, auc_pr_test = metrices(target_list_test, prediction_list_test, probability_list_test, "test")

    print('acc_bur: %3.4f, pear_asa: %3.4f, acc_ppi: %3.4f, AUC_ROC: %3.4f, AUC_PR: %3.4f, prec: %3.4f, recall: %3.4f, TP: %0.1f, FP: %0.1f, TN: %0.1f, FN: %0.1f, time: %3.3f'
        % (np.mean(accuracy_test_bur), np.mean(pearson_test_asa), np.mean(accuracy_test_ppi), auc_roc_test, auc_pr_test, test_precision.result(), test_recall.result(), test_TP.result(), test_FP.result(), test_TN.result(), test_FN.result(), run_time))

    #### ERROR analyse ####
    print("Perform error analyse")
    print(len(filenames_list))
    print(len(target_ppi_list_error))

    for i in range(0,len(filenames_list)):
        print(filenames_list[i])
        # print(target_ppi_list_error[i])
        # print(prediction_ppi_list_error[i])
        # print(len(target_ppi_list_error[i]))
        # print(len(prediction_ppi_list_error[i]))

        fpr, tpr, auc_roc, precision, recall, precision_list, recall_list, fraction_positive, auc_pr = metrices(target_ppi_list_error[i], prediction_ppi_list_error[i], probability_ppi_list_error[i], "None")
        auc_roc_error_list.append(auc_roc)
        auc_pr_error_list.append(auc_pr)
        precision_error_list.append(precision)
        recall_error_list.append(recall)

        acc_ppi = accuracy_score(target_ppi_list_error[i], prediction_ppi_list_error[i])
        acc_ppi_error_list.append(acc_ppi)

        acc_bur = accuracy_score(target_bur_list_error[i], prediction_bur_list_error[i])
        acc_bur_error_list.append(acc_bur)

        pcc_asa = np.corrcoef(pred_asa, tar_asa)[0][1]
        pcc_asa_error_list.append(pcc_asa)

        output_path = output_erroranalyse
        os.makedirs(output_path, exist_ok=True)

        sumfile_path = output_path + "/IFBU_summary.txt"
        with open(sumfile_path, "w") as f:
            writer_sum = csv.writer(f, delimiter='\t')
            writer_sum.writerow(("ID", "acc_ppi", "auc_roc_ppi", "auc_pr_ppi", "precision_ppi", "recall_ppi", "acc_bur", "pcc_asa"))
            writer_sum.writerows(zip(filenames_list, acc_ppi_error_list, auc_roc_error_list, auc_pr_error_list, precision_error_list, recall_error_list, acc_bur_error_list, pcc_asa_error_list))


    for i in range(0, len(filenames_list)):
        list_filename = [filenames_list[i]]*length_seq_batch[i]
        list_aa_seq = sequences_total[i]
        list_tar = map(int, target_ppi_list_error[i])
        list_pred = prediction_ppi_list_error[i]
        list_prob = probability_ppi_list_error[i]

        datafile_path = output_path + "/" + filenames_list[i] + ".txt"
        with open(datafile_path, "w") as f:
            writer_out = csv.writer(f, delimiter='\t')
            writer_out.writerow(("ID", "AA", "PPI_tar", "PPI_pred", "PPI_prob"))
            writer_out.writerows(zip(list_filename, list_aa_seq, list_tar, list_pred, list_prob))

if __name__ == '__main__':
    main()
