# CSCI946-Assignment-3

Data source: Please download all data from the following 2 links, where in link1 the files named
 ```bash

  1.train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv.zip

  2.train_efficientformerv2_s0.snap_dist_in1k.csv.zip 
```
can not be unzipped, so the second link2 will provide the supplement for these 2 files.

Link1: https://uowmailedu-my.sharepoint.com/:u:/g/personal/leiw_uow_edu_au/Ece1YBv-NfxCs0ZZAiTD2H8Bncovq5_NJqWSq6H7Vf5BGQ?e=Ea8Tnp 

Link2: https://mhteaching.its.uow.edu.au/index.php/s/eoLbFKxpqEfKp5k

The dataset we are going to use are as following:
 ```bash
train_data_path = "data/features/train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
val_data_path_1 = "data/features/val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
val_data_path_2 = "data/features/v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv"
```
Instructions:
1. Amend data directories in all codes to fit your own dataset location
2. Run the following file to train the lightgbm classifier and save it to alex_lgbm_model.joblib for further usage. There is a part of codes are commented out which can use to fine tune hyperparameter.
 ```bash
   classifier.py
 ```
3. Run the following file to get permutaion feature importance for both val sets using trained model alex_lgbm_model.joblib.
```bash
  permutaion_importance_generator.py
```
4. Run the following file to get the feature comparison results and visulize the results.
```bash
  comparefeatures.py
```
5. Base on results, we can modify the classifer to increase the accuracy of val2 and the best result we can get is the same with val1. This part has not been done yet.
