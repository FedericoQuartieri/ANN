# Pirate Pain Time-Series Classification

This repository contains our solution to the **AN2DL First Challenge**: predicting a subject’s true pain level (no_pain, low_pain, high_pain) from multivariate time-series data collected from both ordinary people and pirates.

The project combines a carefully designed preprocessing pipeline with a **GRU-based recurrent neural network** trained via **shuffled K-fold cross-validation** and extensive **grid search** over key hyperparameters. The final model achieves a macro F1-score of **0.9567** on the test set.

---

## Project Overview

At a high level, the workflow is:

1. **Load and clean data** from CSV files.
2. **Engineer temporal features** (especially EWMA-based features) and remove degenerate columns.
3. **Handle outliers and scale features** with time-series-aware clipping and Min–Max normalization.
4. **Window the time series** into fixed-length segments.
5. **Train GRU-based models** with weighted cross-entropy loss using K-fold cross-validation.
6. **Run grid search** to select the best configuration.
7. **Save trained models** and generate predictions.

The focus is on **robust preprocessing**, **regularization**, and **careful hyperparameter selection** rather than on highly complex architectures.

---

## Repository Structure

- `ANN.ipynb`  
  Main notebook with the final data exploration, preprocessing, model training, and evaluation.

- `grid_search.ipynb`  
  Notebook used to run and analyze multiple **grid search** experiments over preprocessing and model hyperparameters.

- `Timeseries_Classification.ipynb`  
  Additional time-series modeling experiments and alternative architectures (e.g., CNN + BiGRU).

- `data/`  
  Contains input CSV files:
  - `pirate_pain_train.csv` – training time-series data.
  - `pirate_pain_train_labels.csv` – ground-truth labels for the training set.
  - `pirate_pain_test.csv` – test time-series data.
  - `sample_submission.csv` – template for generating Codabench submissions.

- `models/`  
  Saved model checkpoints and experiment outputs, organized by configuration and split:
  - `n_val_users_55_k_5_epochs_200_window_size_24_stride_4_hidden_layers_2_hidden_size_128_batch_size_256_learning_rate_0.001_dropout_rate_0.3_l2_lambda_0.0001/` – final configuration with per-split `.pt` weights.
  - `trial_0/` – earlier or alternative experimental run.

- `requirements.txt`  
  Python dependencies required to run the notebooks and reproduce the experiments.
  
---

## Acknowledgements

This work was developed as part of the **Advanced Neural Networks and Deep Learning (AN2DL)** course at Politecnico di Milano by the team **“Transformers: rise of the beasts”**:

- Tommaso Marchesini  
- Federico Quartieri  
- Daniele Salvi  
- Giacomo Tessera  

The project is based on the official pirate pain dataset released for the course challenge.
