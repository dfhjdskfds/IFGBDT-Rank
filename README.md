# IFGBDT-Rank

Here we have collected the code to accompany the preprint submitted to Expert Systems With Applications "Toward Fairness-Aware Gradient Boosting Decision Trees for Ranking"  

Python Package:

- "fair_training_ranking_xgb.py": Contains the code to run IFGBDT-Rank. To run IFGBDT-Rank, use the "train_fair_nn" function. Look at the comments regarding its input.

Folders:

- "Synthetic": This folder contains the code to reproduce the synthetic data experiments. Run the "synthetic.ipynb" notebook.

- "German Data": This folder contains the code to reproduce the German credit experiments. 

  * First, run the "german_credit_preprocessing.ipynb notebook to download the data and preprocess it. 
  * To run the GBDT-Rank, project baseline, and random baseline with the hyperparameters in our paper, run 'python german_baseline.py
  * To run the IFGBDT-Rank experiments with the hyperparameters in our paper, run 'python german.py
  
  - "Microsoft’s Learning to Rank": This folder contains the code to reproduce the Microsoft LTR experiments.
    * First, download the data https://www.microsoft.com/en-us/research/project/mslr/ (the MSLR-WEB10K data), and save it to a folder called "original_data".
    * Second, run the 'preprocess_MSLR.ipynb' notebook to preprocess the data.
    * To run the GBDT-Rank, project baseline, and random baseline with the hyperparameters in our paper, run 'python german_baseline.py
    * To run the IFGBDT-Rank experiments with the hyperparameters in our paper, run 'python german.py
