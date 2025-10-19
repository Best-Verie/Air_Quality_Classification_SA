
```markdown
# Air Quality Classification Using Machine Learning

## Project Overview
This project classifies air quality into categories (Good, Moderate, Poor, Hazardous) using environmental sensor data. The goal is to support automated air quality monitoring systems for environmental sustainability in Rwanda and similar developing regions.

## Problem Statement
Air pollution is a critical public health challenge in rapidly urbanizing areas. This project develops machine learning models to automatically classify air quality levels based on pollutant measurements, enabling real-time monitoring and public health warnings.

## Dataset
**Source:** [Air Quality and Pollution Assessment Dataset](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)

**Description:** 
- 5000+ samples with environmental measurements
- 4 air quality categories: Good, Moderate, Poor, Hazardous
- Features include pollutant concentrations (PM2.5, PM10, NO2, etc.) and meteorological data

## What the Code Does

### 1. Data Loading & Preprocessing
- Loads CSV data from Google Drive or local storage
- Handles missing values and encodes target labels
- Splits data into train (70%), validation (15%), and test (15%) sets
- Standardizes features using StandardScaler

### 2. Traditional Machine Learning (5 Experiments)
Implements and evaluates:
- **Logistic Regression** - Baseline linear model
- **Random Forest** - Ensemble of decision trees
- **SVM (RBF)** - Support Vector Machine with non-linear kernel
- **K-Nearest Neighbors (k=7)** - Instance-based learning
- **Gradient Boosting** - Sequential ensemble method

### 3. Deep Learning (5 Experiments)
Builds neural networks with different configurations:
- **Simple NN** - 2 hidden layers (64, 32 neurons)
- **Deep NN** - 3 hidden layers with Batch Normalization
- **NN + L2 Regularization** - Weight penalty for generalization
- **NN + RMSprop** - Alternative optimizer
- **NN + Low Learning Rate** - Slow, careful training

### 4. Model Evaluation & Visualization
Generates comprehensive analysis:
- **Performance comparison charts** (train/val/test accuracy)
- **Learning curves** (accuracy and loss over epochs)
- **Confusion matrices** (error pattern analysis)
- **ROC curves** (per-class discrimination performance)
- **Results table** (CSV export for report)

## Results
- **Best Model:** Deep Neural Network (95.5% test accuracy)
- **Best Traditional ML:** Gradient Boosting (95.2% test accuracy)
- All models achieved >92% accuracy
- Models make reasonable errors only between adjacent air quality categories

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
opencv-python
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python
```

## Usage

### For Google Colab:
1. Upload dataset to Google Drive
2. Mount Drive and update file path in code
3. Run all cells sequentially

### For Local:
1. Download dataset from Kaggle
2. Update `BASE_PATH` variable with your CSV file location
3. Update `target_col` with your target column name
4. Run the notebook



## Project Structure
```
├── air_quality_classification.ipynb    # Main notebook
├── README.md                           # This file
└── report.pdf                          # Academic report
```

## Key Findings
1. Deep learning slightly outperforms traditional ML (0.3% improvement)
2. All approaches achieve excellent accuracy (93-96%)
3. Proper regularization prevents overfitting in neural networks
4. Models understand ordinal nature of air quality (no extreme errors)
5. Simple models (Logistic Regression: 93.5%) provide strong baseline performance

## Applications
- Real-time air quality monitoring systems
- Automated public health alerts
- Environmental policy decision support
- Citizen air quality information services

## Author
Best Verie Iradukunda

## Acknowledgments
- Dataset: Kaggle Air Quality and Pollution Assessment
- Course: Machine Learning Module
- Institution: African Leadership University

## License
This project is for academic purposes.
```
