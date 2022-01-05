# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:09:00 2021

@author: ZW && WCY
"""

from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model
import pandas as pd
import xlrd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
#123,456,789……

codon_table = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'CGU': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R', 'UCU': 'S', 'UCC': 'S',
    'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S', 'AUU': 'I', 'AUC': 'I',
    'AUA': 'I', 'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L',
    'CUG': 'L', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', 'GUU': 'V',
    'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T',
    'ACG': 'T', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'AAU': 'N',
    'AAC': 'N', 'GAU': 'D', 'GAC': 'D', 'UGU': 'C', 'UGC': 'C', 'CAA': 'Q',
    'CAG': 'Q', 'GAA': 'E', 'GAG': 'E', 'CAU': 'H', 'CAC': 'H', 'AAA': 'K',
    'AAG': 'K', 'UUU': 'F', 'UUC': 'F', 'UAU': 'Y', 'UAC': 'Y', 'AUG': 'M',
    'UGG': 'W'
}

def read_excel(name):
    """
    read xlsx data
    :parm name: the xlsx file name
    :retun : data
    """
    book = xlrd.open_workbook(name)
    sheet = book.sheet_by_index(0)
    data = []
    for i in range(0, sheet.nrows):
        data.append(sheet.row_values(i))
    return data

def set_cnn_model_DNA(input_dim=4, input_length=34, nbfilter=50):
    """
    set DNA model
    :parm input_dim: the dimension of input size 
    :parm input_length: the dimension of input length 
    :parm nbfilter: the dimension of filter size 
    :retun model: cnn model 
    """
    model = Sequential()
    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=3,
                     border_mode="valid",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization(name='feature'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

def set_cnn_model_CODON(input_dim=64, input_length=14, nbfilter=50):
    """
    set CODON model
    :parm input_dim: the dimension of input size 
    :parm input_length: the dimension of input length 
    :parm nbfilter: the dimension of filter size 
    :retun model: cnn model 
    """
    model = Sequential()
    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=3,
                     border_mode="valid",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization(name='feature'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

def set_cnn_model_AMINO(input_dim=20, input_length=14, nbfilter=50):
    """
    set AMINO model
    :parm input_dim: the dimension of input size 
    :parm input_length: the dimension of input length 
    :parm nbfilter: the dimension of filter size 
    :retun model: cnn model 
    """
    model = Sequential()
    model.add(Conv1D(input_dim=input_dim, input_length=input_length,
                     nb_filter=nbfilter,
                     filter_length=3,
                     border_mode="valid",
                     subsample_length=1))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization(name='feature'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model

def read_data(path):
    """
    read data
    :param path: the data path
    :return dna: data
    :return label: the label of data
    """
    dna = []
    label = []
    data = read_excel(path)
    for i in range(1,len(data)):
        l = [1,0] if data[i][1] == 'High' else [0,1]
        label.append(l)
        dna.append(data[i][2])
    return dna,label

def get_dna_x(dna):
    """
    segmentation seq
    :param dna: seq
    :parm all_array: the dna seq 
    """
    all_array = []
    for seq in dna:
        alpha = 'ACGT'
        half_len = 2
        row = (len(seq) + 2 * half_len)
        dna_array = np.zeros((row, 4))
        for i in range(half_len):
            dna_array[i] = np.array([0.25] * 4)
            dna_array[-i-1] = np.array([0.25] * 4)
        for i, val in enumerate(seq):
            i = i + half_len
            if val not in 'ACGT':
                dna_array[i] = np.array([0.25] * 4)
                continue
            index = alpha.index(val)
            dna_array[i][index] = 1
        all_array.append(dna_array)
    return all_array

def get_codon_x(dna):
    """
    segmentation seq
    :param dna: dna
    :parm all_array: the codon seq 
    """
    all_array = []
    for seq in dna:
        alpha = 'ACGT'
        half_len = 2
        row = (int(len(seq)/3) + 2 * half_len)
        dna_array = np.zeros((row, 64))
        for i in range(half_len):
            dna_array[i] = np.array([0.01] * 64)
            dna_array[-i-1] = np.array([0.01] * 64)
        for i in range(0,len(seq),3):
            index1 = alpha.index(seq[i])
            index2 = alpha.index(seq[i+1])
            index3 = alpha.index(seq[i+2])
            index = index1 * 16 + index2 * 4 + index3
            dna_array[int(i/3)+half_len][index] = 1
        all_array.append(dna_array)
    return all_array

def get_amino_x(dna):
    """
    segmentation seq
    :param dna: dna
    :parm all_array: the amino seq 
    """
    all_array = []
    for seq in dna:
        seq = seq.replace('T', 'U')
        alpha = 'ACDEFGHIKLMNPQRSTVWY'
        half_len = 2
        row = (int(len(seq)/3) + 2 * half_len)
        dna_array = np.zeros((row, 20))
        for i in range(half_len):
            dna_array[i] = np.array([0.05] * 20)
            dna_array[-i-1] = np.array([0.05] * 20)

        for i in range(0,len(seq),3):
            codon = codon_table[seq[i:i+3]]
            if codon not in alpha:
                dna_array[i] = np.array([0.05] * 20)
                continue
            index = alpha.index(codon)
            dna_array[int(i/3)+half_len][index] = 1
        all_array.append(dna_array)
    return all_array

def cut_train_test(label):
    """
    split train and test data
    :param label: label seq
    :return train_index: training data index
    :return test_index: testing data index
    """
    true_label = []
    false_label = []
    for i in range(0,len(label)):
        if label[i] == [1,0]:
            true_label.append(i)
        else:
            false_label.append(i)
    train_index = true_label[:int(len(true_label)*0.8)]
    train_index.extend(false_label[:int(len(false_label)*0.8)])

    test_index = true_label[int(len(true_label) * 0.8):]
    test_index.extend(false_label[int(len(false_label) * 0.8):])
    return train_index, test_index

def coutResultByExcel(preds, name):
    """
    save data to xlsx
    :param preds: data nees to save
    :param name: the save file name
    """
    preds = pd.DataFrame(preds)
    writer = pd.ExcelWriter(name)
    preds.to_excel(writer, 'page_1', float_format='%.5f',index=False, header=False)  # float_format 控制精度
    writer.save()

def GetLabelFeature(trainFeature, trainLabel,testFeature, testLabel, name):
    """
    save the learned feature
    :param trainFeature: the learned training data feature
    :param trainLabel: the training data label
    :param testFeature: the learned testing data feature
    :param testLabel: the testing data label
    :param name: the save file name
    """
    coutResultByExcel(trainFeature, "result/TrainFeature_model_" + name + ".xlsx")
    coutResultByExcel(trainLabel, "result/TrainLabel_model_" + name + ".xlsx")
    coutResultByExcel(testFeature, "result/ValFeature_model_" + name + ".xlsx")
    coutResultByExcel(testLabel, "result/ValLabel_model_" + name + ".xlsx")

if __name__ == '__main__':
    
    # ----- Read data -----
    dna, label = read_data('data/N-terminal coding sequence-BS.xlsx')

    dna_x = get_dna_x(dna)
    codon_x = get_codon_x(dna)
    amino_x = get_amino_x(dna)

    # ----- Split train and test sets -----
    train_index, test_index = cut_train_test(label)
    train = [i for i in train_index]
    test = [i for i in test_index]
    coutResultByExcel(train, 'result/train_dna.xlsx')
    coutResultByExcel(test, 'result/test_dna.xlsx')

    dna_x_train = np.array([dna_x[i] for i in train_index])
    dna_x_test = np.array([dna_x[i] for i in test_index])

    codon_x_train = np.array([codon_x[i] for i in train_index])
    codon_x_test = np.array([codon_x[i] for i in test_index])

    amino_x_train = np.array([amino_x[i] for i in train_index])
    amino_x_test = np.array([amino_x[i] for i in test_index])

    y_train = np.array([label[i] for i in train_index])
    y_test = np.array([label[i] for i in test_index])

    # ----- Training DNA model -----
    model_dna = set_cnn_model_DNA()
    model_dna.compile(optimizer=Adam(), loss='binary_crossentropy')  # 'mean_squared_error'
    model_dna.fit(dna_x_train, y_train, batch_size=64, nb_epoch=20, verbose=0, class_weight='auto')
    model_dna.save("result/Model_dna.h5", True, True)
    
    # ----- Getting DNA view data -----
    layer_name = 'feature'  
    intermediate_layer_model = Model(inputs=model_dna.input, outputs=model_dna.get_layer(layer_name).output)
    intermediate_output1 = intermediate_layer_model.predict(dna_x_train)
    intermediate_output2 = intermediate_layer_model.predict(dna_x_test)
    GetLabelFeature(intermediate_output1, y_train,intermediate_output2, y_test, "DNA")

    # ----- Training codon model -----
    model_codon = set_cnn_model_CODON()
    model_codon.compile(optimizer=Adam(), loss='binary_crossentropy')  # 'mean_squared_error'
    model_codon.fit(codon_x_train, y_train, batch_size=64, verbose=0, nb_epoch=20, class_weight='auto')
    model_codon.save("result/Model_codon.h5", True, True)
    
    # ----- Getting codon view data -----
    layer_name = 'feature'  
    intermediate_layer_model = Model(inputs=model_codon.input,
                                     outputs=model_codon.get_layer(layer_name).output) 
    intermediate_output1 = intermediate_layer_model.predict(codon_x_train)
    intermediate_output2 = intermediate_layer_model.predict(codon_x_test)
    GetLabelFeature(intermediate_output1, y_train, intermediate_output2, y_test, "CODON")

    # ----- Training amino model -----
    model_amino = set_cnn_model_AMINO()
    model_amino.compile(optimizer=Adam(), loss='binary_crossentropy')  # 'mean_squared_error'
    model_amino.fit(amino_x_train, y_train, batch_size=64, nb_epoch=20, verbose=0, class_weight='auto')
    model_amino.save("result/Model_amino.h5", True, True)
    
    # ----- Getting amino view data -----
    layer_name = 'feature'  
    intermediate_layer_model = Model(inputs=model_amino.input,
                                     outputs=model_amino.get_layer(layer_name).output)
    intermediate_output1 = intermediate_layer_model.predict(amino_x_train)
    intermediate_output2 = intermediate_layer_model.predict(amino_x_test)
    GetLabelFeature(intermediate_output1, y_train, intermediate_output2, y_test, "AMINO")
