# CSCI946-Assignment-3

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
