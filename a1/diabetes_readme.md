# Diabetes Prediction Using Perceptron Algorithm

This project implements and evaluates a Perceptron-based classifier for predicting diabetes using the Pima Indians Diabetes Database. The implementation includes comprehensive data preprocessing, multiple classification algorithms for comparison, and detailed performance analysis.

## Project Overview

This assignment implements a Perceptron algorithm (interpretable as a dense layer neural network) for diabetes prediction. The project includes:

- Data preprocessing and outlier handling
- Single-layer Perceptron implementation
- Comparative analysis with other classifiers (Gaussian Naive Bayes, k-NN, MLP)
- Performance evaluation using multiple metrics (Accuracy, Precision, Sensitivity, F1-Score)
- Visualization of results

## Dataset

The project uses the **Pima Indians Diabetes Database**, which contains diagnostic measurements for predicting diabetes onset.

**Dataset Sources:**
- Original: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- Preprocessed: [LIBSVM Dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)

**Required File:** `diabetes.csv` (should be placed in the project root directory)

### Features
The dataset contains 8 medical predictor variables and 1 target variable:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI (Body Mass Index)
- DiabetesPedigreeFunction
- Age
- Outcome (0 or 1)

## Requirements

### Dependencies

```
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
```

### Python Version
- Python 3.7 or higher

## Installation

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. **Install required packages:**
```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

3. **Download the dataset:**
   - Download `diabetes.csv` from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
   - Place the file in the project root directory

## Project Structure

```
.
├── diabetes.csv              # Dataset (to be downloaded)
├── main.py                   # Main implementation script
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## How to Run

1. **Ensure the dataset is in place:**
   - Verify that `diabetes.csv` is in the project root directory

2. **Run the main script:**
```bash
python main.py
```

3. **Expected Output:**
   - Data preprocessing visualizations (histograms and boxplots for each feature)
   - Classification reports for each model
   - Performance metrics (Accuracy, Precision, Sensitivity, F1-Score)
   - Comparative bar chart of all models

## Implementation Details

### Data Preprocessing
1. **Missing Value Handling:** Zero values in certain features are replaced with median values
2. **Outlier Detection:** IQR method is used to identify and cap outliers
3. **Feature Scaling:** StandardScaler is applied to normalize all features
4. **Train-Test Split:** 80-20 split with stratification

### Models Implemented

1. **Single-layer Perceptron (PPN)**
   - Learning rate: 0.001
   - Max iterations: 5000

2. **Gaussian Naive Bayes (GB)**
   - Baseline probabilistic classifier

3. **k-Nearest Neighbors (k-NN)**
   - Optimal k=4 (determined through cross-validation)

4. **Multi-Layer Perceptron (MLP)**
   - Hidden layer: 12 neurons
   - Activation: ReLU
   - Solver: Adam
   - Learning rate: 0.001

### Performance Metrics

The following metrics are calculated for each model:
- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of positive predictions that are correct
- **Sensitivity (Recall):** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall

## Results

The project outputs:
1. Individual classification reports for each model
2. Comparative performance table
3. Visual comparison bar chart of all metrics across models

## Code Documentation

The implementation includes:
- Comprehensive data exploration and visualization
- Step-by-step feature engineering
- Multiple classifier implementations
- Performance comparison framework

## Notes

- The SkinThickness feature is dropped before MLP training based on feature importance analysis
- All random states are fixed for reproducibility
- The test set is kept separate and only used for final evaluation

## References

- Dataset: UCI Machine Learning Repository - Pima Indians Diabetes Database
- CVPR Format: [Computer Vision and Pattern Recognition Conference](http://cvpr2020.thecvf.com/)
- LIBSVM Tools: [Binary Classification Datasets](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)

## Author

[Your Name]
[Your Email]
[GitHub Repository Link]

## License

This project is submitted as part of an academic assignment.
