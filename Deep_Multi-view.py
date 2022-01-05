# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:09:50 2021

@author: ZW && WCY
"""

import numpy as np
import pandas as pd
import xlrd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score,precision_recall_curve,f1_score,average_precision_score


def read_nsheet_xlsx(name):
    """
    read xlsx data
    :parm name: the xlsx file name
    :retun : data
    """
    book = xlrd.open_workbook(name)
    num_sheet = book.nsheets

    xls = []
    for j in range(0, num_sheet):
        sheet = book.sheet_by_index(j)
        for i in range(0, sheet.nrows):
            xls.append(sheet.row_values(i))
        xls = pd.DataFrame(xls)
        if j == 0:
            data = tuple([xls])
        else:
            data = data +tuple([xls])
        xls = []

    return data

def vec2lab(vec):
    """
    transfrom label matrix from n*c to n*1
    :param vec: n*c lael matrix
    :return : n*1 label matrix
    """
    label = []
    for i in range(0, len(vec)):
        if vec[i,0] > vec[i,1]:
            lab = 1
        else:
            lab = 2
        label.append(lab)

    return label

def lab2vec(lab):
    """
    transfrom label matrix from n*1 to n*c
    :param vec: n*1 lael matrix
    :return : n*c label matrix
    """
    vec = np.zeros((len(lab), 2))
    for i in range(0, len(lab)):
        if lab.iloc[i].values == 1:
            vec[i,0]=1
        else:
            vec[i,1]=1

    return vec

def save_model(preds,name):
    """
    save the model
    :param preds: the learned model
    :param name: the save file name
    """
    view_nums = len(preds)
    writer = pd.ExcelWriter(name)
    for v in range(view_nums):
        preds[v].to_excel(writer, 'page_' + str(v), float_format='%.5f', index=False, header=False)

def train_linear(X, labels):
    """
    inital the single view model
    :param X: the training data n*d n is the number of instances d is the number of feature
    :param labels: the label matrix of traing data n*1
    :return : best_pg the learned model
    """
    data = X.values
    labels_vec = lab2vec(labels)
    labels = labels.values
    results = np.zeros((5,1))
    kf = KFold(n_splits=5,shuffle=True)
    auc_best = 0
    for la in range(-5, 6):
        fold = 0
        
        for train_index, test_index in kf.split(data):
            
            train_X = data[train_index]
            train_labels = labels_vec[train_index.T]
            test_X = data[test_index.T]
            test_labels = labels[test_index]
            lambda1 = pow(2, la)
            Xg1 = np.dot(train_X.T,train_X)

            pg =np.dot( np.linalg.pinv(Xg1 + lambda1*np.identity(Xg1.shape[0])), np.dot(train_X.T, train_labels))
            Y_te = np.dot(test_X, pg)
            lab_te = vec2lab(Y_te)
            
            results[fold] = roc_auc_score(test_labels, lab_te)
            fold = fold + 1

        if auc_best < np.mean(results):
            auc_best = np.mean(results)
            best_pg = pg

    return best_pg

def train_mul_linear(mulview_data_cell, model_cell, view_weight, T, W, options):
    """
    traning multi-view model
    :param mulview_data_cell: multi-view data, this is a cell data view_nums*1, each view data is n*d
    :param model_cell: the initialized model for each view, this is a cell data view_nums*1, each view model is d*c
    :param view_weight: the initialized view weight
    :param T: the label matrix n*c
    :param W: the instance weight matrix n*n, this is a diagonal matrix
    :param options: contains the parameters needed for the model
    
    :return model_cell: the learned best model 
    :return view_weight: the learned best view weight
    
    """
    c = np.size(T,1)
    view_nums = options['view_nums']
    N = len(mulview_data_cell[0])
    lambda1 = options['lambda1']
    lambda2 = options['lambda2']
    lambda3 = options['lambda3']
    maxiter = options['maxiter']
    for i in range(0, maxiter):
        sum_weight = 0
        for view_num in range(0, view_nums):
            temp_pg = model_cell[view_num]
            temp_x = mulview_data_cell[view_num]

            sum_variance = np.linalg.norm( np.dot(W, (np.dot(temp_x,temp_pg) - T)))
            sum_variance = np.exp(-lambda3 * sum_variance)
            sum_weight = sum_weight + sum_variance

        for view_num in range(0, view_nums):
            acc_p = model_cell[view_num]
            x = mulview_data_cell[view_num]
            variance = np.linalg.norm(np.dot(W, (np.dot(x, acc_p)  - T)))
            acc_w = np.exp(-lambda3 * variance) / sum_weight # update view weight
            sum_y = np.zeros((N, c))
            
            for j in range(0, view_nums):
                if j != view_num:
                    temp_pg = model_cell[j]
                    temp_x = mulview_data_cell[j]
                    sum_y = sum_y + np.dot(temp_x, temp_pg)

            y_cooperate = sum_y / (view_nums - 1)
            z = acc_w * np.dot(np.dot(np.dot(x.T, W.T), W), x)
            part_a = np.linalg.pinv(z + lambda1*np.identity(z.shape[0]) + lambda2*np.dot(x.T, x))
            part_b = (acc_w * np.dot(np.dot(np.dot(x.T, W.T),  W.T),T ) + lambda2*np.dot( x.T, y_cooperate))
            if view_num == 0:
                model_cell_new = tuple([pd.DataFrame(np.dot(part_a, part_b))]) # update model
            else:
                model_cell_new = model_cell_new + tuple([pd.DataFrame(np.dot(part_a, part_b))]) # update model
            
            view_weight[view_num] = acc_w
        
        del model_cell
        model_cell = model_cell_new
        del model_cell_new

    return model_cell, view_weight

def test_mul_linear(test_data_cell, model_cell, view_weight, view_nums, c):
    """
    Predict multi-view data
    :param test_data_cell: multi-view data needs to predict,this is a cell data view_nums*1, each view data is n*d
    :param model_cell: the learned multi-view model
    :param view_weight: the learned view weight
    :param view_nums: the number of view number
    :param c: the numer of class
    
    :return Y_te: predicted results
    
    """
    N = len(test_data_cell[0])
    Y_te = np.zeros((N, c))

    for view_num in range(0,view_nums):
        acc_pg = model_cell[view_num]
        acc_w = view_weight[view_num]
        acc_x = test_data_cell[view_num]
        Y_te = Y_te + acc_w*np.dot(acc_x, acc_pg)

    Y_te[Y_te < 0] = 0
    Y_te[Y_te > 1] = 1

    return Y_te

def trapezoid(x1,x2,y1,y2):
    a = np.abs(x1-x2) 
    b =  np.abs(y1+y2) 
    area = a * b / 2
    return area

def roc_curve_zw(score, target,Lp,Ln):
    """
    calculate auc and roc
    """
    length = len(score)
    if length != len(target):
        print('The length of tow input vectors should be equal\n')
        return 
    P = 0
    N = 0
    for i in range(length):
        if target[i] == Lp:
            P = P + 1
        elif target[i] == Ln:
            N = N + 1
    
    L = [score, target]
    L = pd.DataFrame(L)
    L=L.T
    L = L.sort_values(0,ascending = False).reset_index(drop=True)
    L = L.values
    
    fp = 0
    fp_pre = 0
    tp = 0
    tp_pre = 0
    score_pre = -10000
    curve = list([])
    auc = 0
    for i in range(length):
        if L[i,0] != score_pre:
            curve.append([fp/N, tp/P, L[i,0]])
            auc = auc + trapezoid(fp, fp_pre, tp, tp_pre) 
            score_pre = L[i,0]
    
            fp_pre = fp
            tp_pre = tp 
        
        if L[i,1] == Lp:
            tp = tp + 1
        else:
            fp = fp + 1
    
    curve.append([1,1,0])
    curve = pd.DataFrame(curve)
    auc = auc / P / N
    auc = auc + trapezoid(1, fp_pre/N, 1, tp_pre/P)
    
    return auc, curve

def compute_mAP(labels,y_pred):
    y_true = labels.values
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i][0],y_pred[i]))
    return np.mean(AP)


#---------------------- Read data ----------------------

min_max_scaler = preprocessing.MinMaxScaler()

view1 = read_nsheet_xlsx('data/TestFeature_AMINO.xlsx')
view1 = view1[0]
view1 = min_max_scaler.fit_transform(view1)
view1 = pd.DataFrame(view1)

view2 = read_nsheet_xlsx('data/TestFeature_CODON.xlsx')
view2 = view2[0]
view2 = min_max_scaler.fit_transform(view2)
view2 = pd.DataFrame(view2)

view3 = read_nsheet_xlsx('data/TestFeature_DNA.xlsx')
view3 = view3[0]
view3 = min_max_scaler.fit_transform(view3)
view3 = pd.DataFrame(view3)

view4 = read_nsheet_xlsx('data/train-view4.xlsx')
view4 = view4[0]
view4 = min_max_scaler.fit_transform(view4)
view4 = pd.DataFrame(view4)
view4 = view4[[0,1]]



label = np.zeros(5521)
label[0:4902] = 1
label[4902:] = 2
tr_ind = list(range(0,4902))
te_ind = list(range(4902,5521))
label = pd.DataFrame(label)


#label = np.zeros(15579)
#label[0:5231] = 1
#label[5231:] = 2
#tr_ind = list(range(0,5231))
#te_ind = list(range(5231,15579))
#label = pd.DataFrame(label)


#---------------------- Initialization ----------------------
data = tuple([view1, view2, view3, view4, label])

view_nums = len(data)
label = data[view_nums - 1]
label.rename(columns={0:'label'}, inplace=True)

view_nums = view_nums - 1
options = {}
options['view_nums'] = view_nums # view number
options['maxiter'] = 10 # Iteration number
c = max(label.values) # class number
view_weight = np.zeros(view_nums) # view weight


model_cell={}
for view_num in range(0, view_nums):
    pg = train_linear(data[view_num], label) # the initialized model
    if view_num == 0:
        model_cell = tuple([pd.DataFrame(pg)])
    else:
        model_cell = model_cell + tuple([pd.DataFrame(pg)])
    view_weight[view_num] = 1 / view_nums # the initialized view weight

best_auc_te = 0
best_acc_te = 0
best_f1_te = 0
auc = np.zeros(5)
f1 = np.zeros(5)


#---------------------- Training model ----------------------
for lambda1 in range(-5,0):
    for lambda2 in range(-5, 0):
        for lambda3 in range(-5, 0):
            options['lambda1'] = pow(2, lambda1) # regularization parameter
            options['lambda2'] = pow(2, lambda2) # cooperative learning parameter
            options['lambda3'] = pow(2, lambda3) # negative Shannon entropy parameter
            
            for iter in range(5):
                
                # -----split train and valid data-----
                
                train_ind_h, valid_ind_h = model_selection.train_test_split(tr_ind, test_size = 0.2)
                train_ind_l, valid_ind_l = model_selection.train_test_split(te_ind, test_size = 0.2)
                
                train_ind = train_ind_h + train_ind_l
                valid_ind = valid_ind_h + valid_ind_l
                
                train_view1 = view1.iloc[train_ind]
                train_view2 = view2.iloc[train_ind]
                train_view3 = view3.iloc[train_ind]
                train_view4 = view4.iloc[train_ind]
                train_labels = label.values[train_ind]
                train_labels = pd.DataFrame(train_labels)
                train_labels.rename(columns={0:'label'}, inplace=True)
                
                test_view1 = view1.iloc[valid_ind]
                test_view2 = view2.iloc[valid_ind]
                test_view3 = view3.iloc[valid_ind]
                test_view4 = view4.iloc[valid_ind]
                test_labels = label.values[valid_ind]
                test_labels = pd.DataFrame(test_labels)
                test_labels.rename(columns={0:'label'}, inplace=True)
                
                train_data = tuple([train_view1, train_view2, train_view3, train_view4, train_labels])
                test_data = tuple([test_view1, test_view2, test_view3, test_view4, test_labels])
                
                del train_view1, train_view2, train_view3,test_view1, test_view2, test_view3,
                
                N = len(train_data[0])
                W = np.identity(N)
    
                index = train_labels[train_labels.label == 1].index.tolist()
                index_fu = train_labels[train_labels.label == 2].index.tolist()
                
                # ----- set instance weight matrix -----
                if len(index) / len(index_fu) > 1.5:
                    for i in range(0, len(index_fu)):
                        W[index_fu[i], index_fu[i]] = 30
                #        W[index[i], index[i]] = 0.5
                
                if len(index_fu) / len(index) > 1.5:
                    for i in range(0, len(index)):
                        W[index[i], index[i]] = 10
#                        W[index[i], index[i]] = 0.2
            
                # ----- training model -----
                model_cell_t, view_weight_t = train_mul_linear(train_data, model_cell, view_weight, lab2vec(train_labels), W,
                                                             options)
                
                # ----- testing model -----
                Y_te = test_mul_linear(test_data, model_cell_t, view_weight_t, view_nums, int(c[0]))
                labels_te = vec2lab(Y_te)
                acc = accuracy_score(test_labels, labels_te)
                auc[iter], curve = roc_curve_zw(Y_te[:,0],test_labels.values,1,2)
                f1[iter] = f1_score(test_labels, labels_te)
           
            # ----- logging optimal results -----
            if np.mean(auc) > best_auc_te:
                
                best_auc_te = np.mean(auc)
                best_auc = auc

                best_f1 = f1
                best_model = model_cell_t
                best_curve = curve
                best_Y_te = Y_te
                best_labels_te = labels_te
                best_view = view_weight_t
                best_options = options
                print("ACC: %s   AUC: %s  F1: %s" %(acc,auc,f1))


#---------------------- Saving results&model ----------------------

best_curve.to_excel('results_5fold/roc.xlsx')

best_f1 = pd.DataFrame(best_f1)
best_f1.to_excel('results_5fold/f1.xlsx')

best_auc = pd.DataFrame(best_auc)
best_auc.to_excel('results_5fold/auc.xlsx')

save_model(best_model, "results_5fold/best_linear_model.xlsx")
writer = pd.ExcelWriter("results_5fold/best_view_weight.xlsx")
best_view = pd.DataFrame(best_view)
best_view.to_excel(writer, 'page_1', float_format='%.5f', index=False, header=False)
writer.save()




    
    