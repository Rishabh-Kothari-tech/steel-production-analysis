# Steel Quality Prediction using Machine Learning

This repository contains the Applied Machine Learning project for
the course **Applied Machine & Deep Learning (190.015)** at
Montanuniversität Leoben.

Repository Structure

- `src/`
  - `notebooks/` – Text and .json files of output.
  - `scripts/` – Python scripts for data preprocessing, model training, and evaluation.
- `data/`
  - `raw/` – Original input datasets.
  - `processed/` – Cleaned and transformed datasets used for modeling.
- `docs/`
  - `report/` – Project report (PDF).
  - `ppt/` - Project presentation.
- `results/`
  - `figures/` – visualizations of model performance and data analysis.
  - `plots/` - plots of model performance
  - `tables/` – Metrics table.
- `README.md` – This file describing the project and repository layout.

## Project Overview

The goal of this project is to predict a continuous steel quality factor from process data using supervised regression models. The data is normalized process data containing relevant features that influence the final steel quality.

The main objectives are:
- To preprocess and normalize the given process data for model training.
- To train and compare several regression models for steel quality prediction.
- To evaluate model performance using standard regression metrics and select the best-performing approach.

## Methods

### Data Acquisition

- The dataset consists of process variables and target steel quality values provided as part of the course material.
- The raw data are stored in `data/raw/`, while cleaned and feature-engineered versions are stored in `data/processed/`.

### Data Analysis

- Data preprocessing includes handling missing values, normalization, and train–test splitting.
- Multiple regression models are trained and tuned, and their performance is compared using common error metrics and visual diagnostics.
- Visualizations such as error distributions, predicted vs. true plots, are generated and stored in `results/figures/`.

### Models Used

The following supervised regression models are implemented and compared:

- Random Forest Regressor
- Support Vector Regression (SVR)  
- Multi layor Perceptron (MLP)
- Gaussian Process Regressor


### Tools Used

- Programming language: Python  
- Libraries: scikit-learn, pandas, numpy, matplotlib/seaborn 
- Environment: Visual Studio Code for experimentation, Python scripts for reproducible runs.

## Results

### Evaluation Metrics

Model performance is evaluated using the following metrics:

- Mean Absolute Error (MAE)    
- Root Mean Squared Error (RMSE)  
- Coefficient of Determination (R² Score)  

### Findings

- Each model’s performance is summarized in metrics tables saved under `results/tables/`.
- Key plots such as predicted vs. actual steel quality and residual plots are stored under `results/figures/` and referenced in the report.

### Visualizations

Examples of visual outputs include:

- Model comparison bar charts of MAE, RMSE, and R².
- Scatter plots of predicted vs. true steel quality values.
- Feature importance plots for tree-based models.

## Files
- `data/` – training and test datasets
- `notebook/` – Jupyter notebook with full implementation
- `report/` – project report (PDF)

## Author
Rishabh Kothari