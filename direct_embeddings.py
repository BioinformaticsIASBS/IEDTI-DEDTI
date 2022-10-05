# requirements
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle 
from scipy import spatial

def save_direct_embeddings(path):

  # reading data
  drugFiles = [
      path+ '/mat_drug_protein.txt',
      path+ '/mat_drug_drug.txt',
      path+ '/mat_drug_disease.txt',
      path+ '/mat_drug_se.txt',
      path+ '/Similarity_Matrix_Drugs.txt']

  proteinFiles = [
      path+ '/mat_protein_drug.txt',
      path+ '/mat_protein_protein.txt',
      path+ '/mat_protein_disease.txt',
      path+ '/Similarity_Matrix_Proteins.txt',
  ]

  drugProtein = pd.read_csv(drugFiles[0], delimiter = ' ', header=None).to_numpy()
  drug_drug = pd.read_csv(drugFiles[1], delimiter = ' ', header=None).to_numpy()
  drug_disease = pd.read_csv(drugFiles[2], delimiter = ' ', header=None).to_numpy()
  drug_se = pd.read_csv(drugFiles[3], delimiter = ' ', header=None).to_numpy()
  similarity_matrix_drug = pd.read_csv(drugFiles[4], delimiter = '    ', header=None).to_numpy()

  protein_drug = pd.read_csv(proteinFiles[0], delimiter = ' ', header=None).to_numpy()
  protein_protein = pd.read_csv(proteinFiles[1], delimiter = ' ', header=None).to_numpy()
  protein_disease = pd.read_csv(proteinFiles[2], delimiter = ' ', header=None).to_numpy()
  Similarity_Matrix_Proteins = pd.read_csv(proteinFiles[3], delimiter = ' ', header=None).to_numpy()

  # save indexes in arrays
  Y = []
  XIndex = []
  for i in range(len(drugProtein)):
      for j in range(len(drugProtein[0])):

          Y.append(
                drugProtein[i][j]
          )

          XIndex.append(
              [
                  i, j                 
              ]  
          )

  Y = np.array(Y)

  # shuffle arrays of indexes
  XIndex, Y = shuffle(XIndex, Y, random_state=0)

  # save indexes of data and labels in seprate files
  with open('XIndex.npy', 'wb') as f:
      np.save(f, np.array(XIndex))

  with open('Y.npy', 'wb') as f:
      np.save(f, np.array(Y))


  # function to find similartites 
  def find_similarity_matrix(input_matrix):
    input_matrix_len = len(input_matrix)
    similarity_matrix = np.zeros((input_matrix_len, input_matrix_len))

    for i in range(input_matrix_len):
      for j in range(input_matrix_len):
        i_j_rows_similarity = 1 - spatial.distance.cosine(input_matrix[i], input_matrix[j])
        similarity_matrix[i][j] = i_j_rows_similarity

    return similarity_matrix


  # function for mixing different similarity matrixes
  def mix_matrix(input_matrixes, matrix_size, operation_name):

    number_of_input_matrixes = len(input_matrixes)

    summed_matrix = np.zeros((matrix_size, matrix_size))
    max_matrix = np.zeros((matrix_size, matrix_size))
    min_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
      for k in range(matrix_size):
        
        temp_sum = 0
        temp_max = 0
        temp_min = 0

        if operation_name == 'sum':
          for j in range(number_of_input_matrixes):
            temp_sum = temp_sum + input_matrixes[j][i][k]
            summed_matrix[i][k] = temp_sum
          
        if operation_name == 'max':
          for j in range(number_of_input_matrixes):
            temp_max = np.maximum(temp_max, input_matrixes[j][i][k])
            max_matrix[i][k] = temp_max

        if operation_name == 'min':
          for j in range(number_of_input_matrixes):
            temp_min = np.minimum(temp_max, input_matrixes[j][i][k])
            min_matrix[i][k] = temp_min     

    if operation_name == 'sum':
      return summed_matrix
    if operation_name == 'max':
      return max_matrix
    if operation_name == 'min':
      return min_matrix


  # find similarities for drugs
  # similarity between drug and se, protein, drug, and di 
  similarity_matrix_drug_se = find_similarity_matrix(drug_se)
  similarity_matrix_drugProtein  = find_similarity_matrix(drugProtein)
  similarity_matrix_drug_disease = find_similarity_matrix(drug_disease)

  # get direct embeddings of drugs and save it
  mixed_drugs = mix_matrix([similarity_matrix_drug_se, similarity_matrix_drug_disease, similarity_matrix_drug, drug_drug], 708, 'sum')
  np.savetxt(fname=path+ 'mixed_drug_se_disease_drug_4matrix_708size.txt' , X = mixed_drugs)

  # find similarities for proteins 
  # similarity between protein and disease 
  similarity_matrix_protein_disease = find_similarity_matrix(protein_disease)

  # get direct embeddings of proteins and save it
  mixed_proteins = mix_matrix([similarity_matrix_protein_disease, protein_protein, Similarity_Matrix_Proteins],1512 , 'sum')
  np.savetxt(fname=path+ '/mixed_protein_disease_protein_3matrix_1512size.txt' , X = mixed_proteins)

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser() #description="training data")
  parser.add_argument('--data_path', type=str, required=True)

  args = parser.parse_args()
  config = vars(args)
  print(config)

  save_direct_embeddings(args.data_path)
