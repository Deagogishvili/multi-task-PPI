# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang
@editor: Henriette Capel (05-02-2021)

"""

import os
import tensorflow as tf
from my_transformer import Transformer, create_padding_mask
from my_rnn_prob import BiRNN_IF, BiRNN_S8, BiRNN_S3, BiRNN_SA, BiRNN_BU, BiRNN_IFBU, BiRNN_IFS3, BiRNN_IFS8, BiRNN_IFSA, BiRNN_IFPP, BiRNN_IFBUS3, BiRNN_IFBUS8, BiRNN_IFBUSA, BiRNN_IFBUPP, BiRNN_IFS3S8, BiRNN_IFS3SA, BiRNN_IFS8SA, BiRNN_IFBUSAPP, BiRNN_IFBUS3S8, BiRNN_IFSAS3S8, BiRNN_IFBUS3SA, BiRNN_IFBUS8SA, BiRNN_IFBUS3S8SA, BiRNN_IFBUS3SAPP, BiRNN_IFBUS8SAPP, BiRNN_IFBUS3S8SAPP
from my_cnn import CNN
from utils_mylabels_prob import clean_inputs, compute_cross_entropy_loss, compute_mse_loss, compute_cross_entropy_loss_ppi

class Model(object):

    def __init__(self, params, name):

        self.params = params
        self.name = name

        self.transformer = Transformer(num_layers=self.params["transfomer_layers"],
                                       d_model=self.params["d_input"],
                                       num_heads=self.params["transfomer_num_heads"],
                                       rate=self.params["dropout_rate"])

        self.cnn = CNN(num_layers=self.params["cnn_layers"],
                       channels=self.params["cnn_channels"])

        if self.name == "IF":
            self.birnn = BiRNN_IF(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"])
            print ("use IF model...")

        if self.name == "S8":
            self.birnn = BiRNN_S8(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss8_output=self.params["d_ss8_output"])
            print ("use S8 model...")

        if self.name == "S3":
            self.birnn = BiRNN_S3(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ss3_output=self.params["d_ss3_output"])
            print ("use S3 model...")

        if self.name == "SA":
            self.birnn = BiRNN_SA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  asa_output=self.params["d_asa_output"])
            print ("use SA model...")

        if self.name == "BU":
            self.birnn = BiRNN_BU(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  buried_output=self.params["d_buried_output"])
            print ("use BU model...")

        elif self.name == "IFBU":
            self.birnn = BiRNN_IFBU(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"])
            print ("use IF-BU model...")

        elif self.name == "IFS3":
            self.birnn = BiRNN_IFS3(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  ss3_output=self.params["d_ss3_output"])
            print ("use IF-S3 model...")

        elif self.name == "IFS8":
            self.birnn = BiRNN_IFS8(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  ss8_output=self.params["d_ss8_output"])
            print ("use IF-S8 model...")

        elif self.name == "IFSA":
            self.birnn = BiRNN_IFSA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-SA model...")

        elif self.name == "IFPP":
            self.birnn = BiRNN_IFPP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use IF-PhiPsi model...")

        elif self.name == "IFBUS3":
            self.birnn = BiRNN_IFBUS3(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss3_output=self.params["d_ss3_output"])
            print ("use IF-BU-S3 model...")

        elif self.name == "IFBUS8":
            self.birnn = BiRNN_IFBUS8(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss8_output=self.params["d_ss8_output"])
            print ("use IF-BU-S8 model...")

        elif self.name == "IFBUSA":
            self.birnn = BiRNN_IFBUSA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-BU-SA model...")

        elif self.name == "IFBUPP":
            self.birnn = BiRNN_IFBUPP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use IF-BU-PP model...")

        elif self.name == "IFS3S8":
            self.birnn = BiRNN_IFS3S8(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  ss8_output=self.params["d_ss8_output"])
            print ("use IF-S3-S8 model...")

        elif self.name == "IFS3SA":
            self.birnn = BiRNN_IFS3SA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-S3-SA model...")

        elif self.name == "IFS8SA":
            self.birnn = BiRNN_IFS8SA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  ss8_output=self.params["d_ss8_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-S8-SA model...")


        elif self.name == "IFBUS3S8":
            self.birnn = BiRNN_IFBUS3S8(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  ss8_output=self.params["d_ss8_output"])
            print ("use IF-BU-S3-S8 model...")

        elif self.name == "IFSAS3S8":
            self.birnn = BiRNN_IFSAS3S8(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  asa_output=self.params["d_asa_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  ss8_output=self.params["d_ss8_output"])
            print ("use IF-SA-S3-S8 model...")

        elif self.name == "IFBUS3SA":
            self.birnn = BiRNN_IFBUS3SA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-BU-S3-SA model...")

        elif self.name == "IFBUS8SA":
            self.birnn = BiRNN_IFBUS8SA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss8_output=self.params["d_ss8_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-BU-S8-SA model...")

        elif self.name == "IFBUSAPP":
            self.birnn = BiRNN_IFBUSAPP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  asa_output=self.params["d_asa_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use IF-BU-SA-PP model...")

        elif self.name == "IFBUS3S8SA":
            self.birnn = BiRNN_IFBUS3S8SA(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  ss8_output=self.params["d_ss8_output"],
                                  asa_output=self.params["d_asa_output"])
            print ("use IF-BU-SA-S3-S8 model...")

        elif self.name == "IFBUS3SAPP":
            self.birnn = BiRNN_IFBUS3SAPP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  asa_output=self.params["d_asa_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use IF-BU-S3-SA-PP model...")

        elif self.name == "IFBUS8SAPP":
            self.birnn = BiRNN_IFBUS8SAPP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss8_output=self.params["d_ss8_output"],
                                  asa_output=self.params["d_asa_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use IF-BU-S8-SA-PP model...")

        elif self.name == "IFBUS3S8SAPP":
            self.birnn = BiRNN_IFBUS3S8SAPP(num_layers=self.params["lstm_layers"],
                                  units=self.params["lstm_units"],
                                  rate=self.params["dropout_rate"],
                                  ppi_output=self.params["d_ppi_output"],
                                  buried_output=self.params["d_buried_output"],
                                  ss3_output=self.params["d_ss3_output"],
                                  ss8_output=self.params["d_ss8_output"],
                                  asa_output=self.params["d_asa_output"],
                                  phipsi_output=self.params["d_phipsi_output"])
            print ("use IF-BU-SA-S3-S8-PP model...")



    def inference(self, x, x_mask, y, y_mask, training):

        encoder_padding_mask = create_padding_mask(x_mask)

        x = clean_inputs(x, x_mask, self.params["d_input"])

        transformer_out = self.transformer(x, encoder_padding_mask, training=training)
        cnn_out = self.cnn(x, training=training)
        x = tf.concat((x, cnn_out, transformer_out), -1)

        x = clean_inputs(x, x_mask, 3*self.params["d_input"])

        if self.name == "IF":
            ppi_predictions = \
                self.birnn(x, x_mask, training=training)
            loss = None
            if training == True:
                loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
            return ppi_predictions, loss

        elif self.name == "S8":
            ss8_predictions = \
                self.birnn(x, x_mask, training=training)
            loss = None
            if training == True:
                loss = self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
            return ss8_predictions, loss

        elif self.name == "S3":
            ss3_predictions = \
                self.birnn(x, x_mask, training=training)
            loss = None
            if training == True:
                loss = self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33])
            return ss3_predictions, loss

        elif self.name == "SA":
            asa_predictions = \
                self.birnn(x, x_mask, training=training)
            loss = None
            if training == True:
                loss = self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return asa_predictions, loss

        elif self.name == "BU":
            buried_predictions = \
                self.birnn(x, x_mask, training=training)
            loss = None
            if training == True:
                loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35])
            return buried_predictions, loss

        elif self.name == "IFBU":
            ppi_predictions, buried_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35])

            return ppi_predictions, buried_predictions, loss

        elif self.name == "IFS3":
            ppi_predictions, ss3_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33])
            return ppi_predictions, ss3_predictions, loss

        elif self.name == "IFS8":
            ppi_predictions, ss8_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
            return ppi_predictions, ss8_predictions, loss

        elif self.name == "IFSA":
            ppi_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, asa_predictions, loss

        elif self.name == "IFPP":
            ppi_predictions, phipsi_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_phipsi"]*compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_phipsi"]*compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return ppi_predictions, phipsi_predictions, loss


        elif self.name == "IFBUS3":
            ppi_predictions, buried_predictions, ss3_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33])
            return ppi_predictions, buried_predictions, ss3_predictions, loss


        elif self.name == "IFBUS8":
            ppi_predictions, buried_predictions, ss8_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
            return ppi_predictions, buried_predictions, ss8_predictions, loss

        elif self.name == "IFBUSA":
            ppi_predictions, buried_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, buried_predictions, asa_predictions, loss

        elif self.name == "IFBUPP":
            ppi_predictions, buried_predictions, phipsi_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return ppi_predictions, buried_predictions, phipsi_predictions, loss

        elif self.name == "IFS3S8":
            ppi_predictions, ss3_predictions, ss8_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
            return ppi_predictions, ss3_predictions, ss8_predictions, loss


        elif self.name == "IFS3SA":
            ppi_predictions, ss3_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, ss3_predictions, asa_predictions, loss

        elif self.name == "IFS8SA":
            ppi_predictions, ss8_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, ss8_predictions, asa_predictions, loss

        elif self.name == "IFBUS3S8":
            ppi_predictions, buried_predictions, ss3_predictions, ss8_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
            return ppi_predictions, buried_predictions, ss3_predictions, ss8_predictions, loss

        elif self.name == "IFSAS3S8":
            ppi_predictions, asa_predictions, ss3_predictions, ss8_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8])
            return ppi_predictions, asa_predictions, ss3_predictions, ss8_predictions, loss

        elif self.name == "IFBUS3SA":
            ppi_predictions, buried_predictions, ss3_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, buried_predictions, ss3_predictions, asa_predictions, loss


        elif self.name == "IFBUS8SA":
            ppi_predictions, buried_predictions, ss8_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, buried_predictions, ss8_predictions, asa_predictions, loss

        elif self.name == "IFBUSAPP":
            ppi_predictions, buried_predictions, asa_predictions, phipsi_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])

            return ppi_predictions, buried_predictions, asa_predictions, phipsi_predictions, loss

        elif self.name == "IFBUS3S8SA":
            ppi_predictions, buried_predictions, ss3_predictions, ss8_predictions, asa_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1))
            return ppi_predictions, buried_predictions, ss3_predictions, ss8_predictions, asa_predictions, loss

        elif self.name == "IFBUS3SAPP":
            ppi_predictions, buried_predictions, ss3_predictions, asa_predictions, phipsi_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return ppi_predictions, buried_predictions, ss3_predictions, asa_predictions, phipsi_predictions, loss


        elif self.name == "IFBUS8SAPP":
            ppi_predictions, buried_predictions, ss8_predictions, asa_predictions, phipsi_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return ppi_predictions, buried_predictions, ss8_predictions, asa_predictions, phipsi_predictions, loss


        elif self.name == "IFBUS3S8SAPP":
            ppi_predictions, buried_predictions, ss3_predictions, ss8_predictions, asa_predictions, phipsi_predictions = self.birnn(x, x_mask, training=training)

            loss = None
            if training == True:
                loss_ppi = compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights'])
                is_nan = tf.math.is_nan(loss_ppi)

                if is_nan:
                    #loss only determined by buried
                    loss = self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
                else:
                    loss = self.params["weight_loss_ppi"]*compute_cross_entropy_loss_ppi(ppi_predictions, tf.expand_dims(y[:,:,35], -1), tf.expand_dims(y_mask[:,:,35], -1), self.params['class_weights']) + \
                        self.params["weight_loss_buried"] * compute_cross_entropy_loss(buried_predictions, y[:,:,33:35], y_mask[:,:,33:35]) + \
                        self.params["weight_loss_ss3"] * compute_cross_entropy_loss(ss3_predictions, y[:,:,30:33], y_mask[:,:,30:33]) + \
                        self.params["weight_loss_ss8"] * compute_cross_entropy_loss(ss8_predictions, y[:,:,:8], y_mask[:,:,:8]) + \
                        self.params["weigth_loss_asa"] *compute_mse_loss(asa_predictions, tf.expand_dims(y[:,:,23],-1), tf.expand_dims(y_mask[:,:,23],-1)) + \
                        self.params["weight_loss_phipsi"] * compute_mse_loss(phipsi_predictions, y[:,:,11:15], y_mask[:,:,11:15])
            return ppi_predictions, buried_predictions, ss3_predictions, ss8_predictions, asa_predictions, phipsi_predictions, loss

    def save_model(self):
        print ("save model:", self.name)
        self.transformer.save_weights(os.path.join(self.params["save_path"], self.name + '_trans_model_weight'))
        self.cnn.save_weights(os.path.join(self.params["save_path"], self.name + '_cnn_model_weight'))
        self.birnn.save_weights(os.path.join(self.params["save_path"], self.name + '_birnn_model_weight'))

    def load_model(self):
        print ("load model:", self.name)
        self.transformer.load_weights(os.path.join(self.params["save_path"], self.name + '_trans_model_weight'))
        self.cnn.load_weights(os.path.join(self.params["save_path"], self.name + '_cnn_model_weight'))
        self.birnn.load_weights(os.path.join(self.params["save_path"], self.name + '_birnn_model_weight'))
