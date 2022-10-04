# requirements
import argparse
import sys
from sklearn.model_selection import KFold
from keras.models import load_model
import numpy as np
import pandas as pd
from scipy import spatial
import tensorflow as tf
from tensorflow.keras import layers, Model
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, recall_score, average_precision_score, \
    confusion_matrix   
from sklearn.utils import shuffle 


def train_iedti_model(path, ratio, result_path):
  
  #Reading data

  protein_representations = np.loadtxt(path+'/protein_representations.txt')
  drug_representations = np.loadtxt(path+'/drug_representations.txt')

  print('np.shape(protein_representations): ', np.shape(protein_representations))
  print('np.shape(drug_representations): ', np.shape(drug_representations))
  print('\n')


  # --------------- split by ratio ---------------------

  # function to split data regarding ratio
  def ratio_zero_one_index_func(one_labels_index, zero_labels_index, ratio_value):
      zero_labels_index_with_ratio = zero_labels_index[: (len(one_labels_index)* ratio_value)]

      print("len of ones: ", len(one_labels_index), 'len of zeros that we chose:', len(zero_labels_index_with_ratio))
      print('ratio= ', len(zero_labels_index_with_ratio)/len(one_labels_index) )
      return zero_labels_index_with_ratio


  with open(path+'/Y.npy', 'rb') as f:
      Y = np.load(f)

  with open(path+'/XIndex.npy', 'rb') as f:
      XIndex = np.load(f)

  XIndex = np.array(XIndex)
  print('the lenth of train set before applying ratio: ', len(XIndex))


  zero_labels_index = [XIndex[i] for i in range(len(Y)) if Y[i]==0 ]
  one_labels_index = [XIndex[i] for i in range(len(Y)) if Y[i]==1]


  # split data regarding choosen ratio 
  zero_labels_index_ratio = ratio_zero_one_index_func(one_labels_index,zero_labels_index, ratio )

  XIndex = one_labels_index + zero_labels_index_ratio
  Y =  np.array([1 for i in range(len(one_labels_index))] + [0 for i in range(len(zero_labels_index_ratio))])

  print('the lenth of train indexes after applying ratio: ', len(XIndex))
  # print('the lenth of test indexes after applying ratio: ', len(Y))

  # shuffle before splitting the data
  XIndex, Y = shuffle(XIndex, Y)
  np.savetxt(fname=path+'/XIndex_ratio'+str(ratio)+'_triplet.txt' , X = XIndex)
  np.savetxt(fname=path+'/Y_ratio'+str(ratio)+'_triplet.txt' , X = Y)

  # ---------------------------------------



  def buildModule(inputShape,  output_bias=None):
      drop_out_rate = 0.5
      inputLayer = layers.Input(shape = inputShape)
      
      x = layers.Conv1D(128,3, activation = 'relu')(inputLayer)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Conv1D(64,3, activation = 'relu')(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Conv1D(32,3, activation = 'relu')(x)
      x = layers.BatchNormalization()(x)
      x = layers.Dropout(0.5)(x)
      # x = layers.GlobalMaxPooling1D()(x)
      x = layers.Flatten()(x)
      
      x = layers.Dense(128, activation = 'relu')(x)
      x = layers.Dropout(0.5)(x)
      x = layers.Dense(1, activation = 'sigmoid', bias_initializer=tf.keras.initializers.Constant(output_bias))(x)

      model = Model(inputLayer, x)
      # optimizer = tf.keras.optimizers.Adam()
      METRICS = [
        
  #      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  #      tf.keras.metrics.Precision(name='precision'),
  #      tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      ]
    # optimizer = get_optimizer()
      model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

      return model

  # K-fold Cross Validator 
  kfold = KFold(n_splits = 10, shuffle=True)
  fold_no = 1


  auc_roc_test_list = []


  for train_index, test_index in kfold.split(XIndex):
    
    
    X_testIndex= np.array( [XIndex[i] for i in test_index] )
    Y_test =  np.array([Y[i] for i in test_index] )


    X_trainIndex= np.array([XIndex[i] for i in train_index])
    Y_train = np.array([Y[i] for i in train_index])


    # count number of negative samples
    neg_counter = 0
    for i in range(len(Y_train)):
      if Y_train[i] == 0:
        neg_counter +=1
    neg_counter

    #number of negative and positive samples
    neg = neg_counter
    pos = len(Y_train) - neg
    initial_bias = np.log([pos/ neg])

    
    model = buildModule((256+256,1), initial_bias)

    batch_size = 64
    epochs = 1000

    # fold number 
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # initial value for maximom aupr
    max_aupr = 0

    for epoch in range(epochs):
        
        
        lossList = []
        iterates = len(Y_train) // batch_size
        
        

        for it in range(iterates):

            indexRange = [it * batch_size, (it + 1) * batch_size]
            tempX = []
            for cIT in range(indexRange[0], indexRange[1]):

                index = X_trainIndex[cIT]
                #Concat DrugSim(index[0]) and ProteinSim(index[1]) => tempX
          
                i_th_drug_index = index[0]
                j_th_protein_index = index[1]

                #i_th_drug= similarity_matrix_drug_disease[i_th_drug_index]
                i_th_drug= drug_representations[i_th_drug_index]
                j_th_protein = protein_representations[j_th_protein_index]
                
                tempX.append(np.concatenate((i_th_drug , j_th_protein )))

            
            #print('shape of tempx:' , np.shape(tempX))

            tempX = tf.expand_dims(tempX, -1)
                
            loss = model.train_on_batch(
                tempX,
                Y_train[it * batch_size:(it + 1) * batch_size]
            )

            lossList.append(
                loss
            )
            
        print('Train:')
        print('epoch ', epoch)
        lossList_0 = [lossList[i][0] for i in range(len(lossList))]
        lossList_1 = [lossList[i][1] for i in range(len(lossList))]
        lossList_5 = [lossList[i][2] for i in range(len(lossList))]
        print('loss:', np.mean(lossList_0), 'auc:',np.mean(lossList_1), 'aupr:',np.mean(lossList_5))


        print('evaluation:')

        testIT = len(X_testIndex) // batch_size
        
        testRes = []
        testY = []
        for it in range(testIT):
            indexRange = [it * batch_size, (it + 1) * batch_size]
            tempX = []
            for cIT in range(indexRange[0], indexRange[1]):
                index = X_testIndex[cIT]
                testY.append(
                    Y_test[cIT]
                )
                # Concat DrugSim(index[0]) and ProteinSim(index[1]) => tempX
        
                i_th_drug_index = index[0]
                j_th_protein_index = index[1]

                #i_th_drug= similarity_matrix_drug_disease[i_th_drug_index]
                i_th_drug= drug_representations[i_th_drug_index]
                j_th_protein = protein_representations[j_th_protein_index]

                tempX.append(np.concatenate((i_th_drug , j_th_protein )))


            #print('shape of tempx:' , np.shape(tempX))

            tempX = tf.expand_dims(tempX, -1)

            res = model.predict(
                tempX
            )
                
            for iTemp in range(len(res)):
                testRes.append(res[iTemp])
        
        # print('len temp x: ' , len(tempX))
        
        nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(testY, testRes)
        auc_keras = auc(nn_fpr_keras, nn_tpr_keras)

        precision, recall, thresholds = precision_recall_curve(testY, testRes)
        aupr_keras = auc(recall, precision)
        
        #pyplot.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='auc')
        # axis labels
        #pyplot.xlabel('False Positive Rate')
        #pyplot.ylabel('True Positive Rate')
        # show the legend
        #pyplot.legend()
        # show the plot
        #pyplot.show()
        #pyplot.savefig( 'plots/IEDTI'+str(ratio)+'_10fold/' + 'Epoch' + str(epoch)+ 'Fold' + str(fold_no)+ 'auc.png')
        
        #pyplot.plot(recall, precision, marker='.', label='AUPR')
        # axis labels
        #pyplot.xlabel('Recall')
        #pyplot.ylabel('Precision')
        # show the legend
        #pyplot.legend()
        # show the plot
        #pyplot.show()
        #pyplot.savefig( 'plots/IEDTI_ratio'+str(ratio)+'_10fold/' + 'Epoch' + str(epoch)+ 'Fold' + str(fold_no)+  'aupr.png')
        #pyplot.clf()
        
        auc_roc_test_list.append([fold_no, epoch, auc_keras , aupr_keras])
        print('fold number:', fold_no, 'epoch: ' , epoch, 'auc: ', auc_keras, 'aupr: ', aupr_keras)

        np.savetxt(result_path +'/IEDTI'+str(ratio)+'_10fold.csv', auc_roc_test_list, delimiter=',', fmt='%s')
        
        if aupr_keras > max_aupr:
            model.save(result_path + '/IEDTI'+str(ratio)+'_10fold'+str(fold_no)+'.h5')
            #np.savetxt(fname='testRes_triplet_fold'+ str(fold_no) + '_ratio'+str(ratio)+'.txt' , X = testRes)
            #np.savetxt(fname='testY_triplet_fold'+ str(fold_no) + '_ratio'+str(ratio)+'.txt' , X = testY)
            max_aupr = aupr_keras
        
    fold_no = fold_no + 1

    print('max of aupr in fold ' + str(fold_no) + ' is:', max_aupr)



if __name__ == "__main__":
  
  parser = argparse.ArgumentParser() #description="training data")
  parser.add_argument('--data_path', type=str, required=True)
  parser.add_argument('--ratio', type=int, required= True)
  parser.add_argument('--result_path', type=str, required= True)

  args = parser.parse_args()
  config = vars(args)
  print(config)

  train_iedti_model(args.data_path, args.ratio, args.result_path)

