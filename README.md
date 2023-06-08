# IEDTI-DEDTI

This [article](https://doi.org/10.1038/s41598-023-36438-0) proposes two predictive methods of drug-target interactions. It focuses on three aims. 
-	First, this paper criticizes using inner product-based matrix factorization to predict drug-target interactions (DTIs). Matrix factorization is a linear operation and suffers from some drawbacks. We mention some of these drawbacks, which make matrix factorization incapable of correctly predicting drug-target interaction. We explain that matrix factorization cannot locate more complex and nonlinear relationships among drugs and targets. 

-	The second aim of this research is to provide dense representations of drugs and targets. 
 
-	The final aim of this work is to develop two efficient and accurate computational methods (IEDTI and DEDTI) for DTI prediction. In addition to DTI prediction, IEDTI produces embeddings of drugs and targets to have meaningful representations of the objects and more efficient computation. Both methods utilize the deep neural network to predict DTIs best.



# Train IEDTI and DEDTI models
We used Python 3.7.14 and CUDA Version: 11.2 in our experiments.

#### Obtain *direct* embeddings
To obtain direct embeddings you should run command in terminal as follows:
```bash
python direct_embeddings.py --data_path data_folder_name
```

Example:
```bash
python direct_embeddings.py --data_path 'Data'
```
Please note that the embeddings will be saved inside your dataset folder.

#### Obtain *indirect* embeddings

Kmeans can be applied considering different numbers as K. If you want to see the result of choosing different numbers as K with details of each clusters' members:

```bash
python indirect_embeddings.py --data_path data_folder_name --num_of_protein_clusters n --num_of_drug_clusters m --find_best_k True
```
else:

```bash
python indirect_embeddings.py --data_path data_folder_name --num_of_protein_clusters n --num_of_drug_clusters m
```

where n is the int number of protein clusters and m is the int number of drug clusters.
 
 
#### Train IEDTI
Please note that the ratio number (which is 3) can be changed into any other number regarding your research purpose.
```bash
python IEDTI.py --data_path data_folder_name --ratio 3 --result_path results_folder_name
```


#### Train DEDTI

```bash
python DEDTI.py --data_path data_folder_name --ratio 3 --result_path results_folder_name
```

### Citation

```bash
@article{zabihian2023dedti,
  title={DEDTI versus IEDTI: efficient and predictive models of drug-target interactions.},
  author={Zabihian, A., Sayyad, F.Z., Hashemi, S.M. et al.},
  year={2023}
  doi = { Sci Rep 13, 9238 (2023). https://doi.org/10.1038/s41598-023-36438-0}
}
```
