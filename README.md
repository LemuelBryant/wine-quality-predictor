# Wine Quality Prediction

An end-to-end data science project using **scikit-learn** to predict wine quality from physicochemical properties. This project uses the Wine Quality dataset from the UCI Machine Learning Repository to explore data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model development, and model evaluation.

## Project Status

**Current Stage:** Initial project setup

Completed:

* Created project directory structure
* Initialized Git repository
* Configured Python virtual environment
* Installed core project dependencies
* Downloaded the Wine Quality datasets (red and white wine)
* Added project documentation and Git configuration files
* Load and inspect the datasets

Future milestones:

* Perform exploratory data analysis (EDA)
* Clean and preprocess the data
* Engineer features
* Train baseline machine learning models
* Compare model performance
* Tune hyperparameters
* Evaluate the final model
* Deploy the model with a web application (if appropriate)

## Technologies

The project currently uses:

* Python
* NumPy 2.0.1
* pandas 2.2.2
* Matplotlib
* Seaborn
* scikit-learn
* Jupyter Notebook

Additional packages such as **joblib** and **Streamlit** will be added later as they become necessary during model development and deployment.

## Project Structure

```text
wine-quality-predictor/
│
├── app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

This project uses the **Wine Quality** dataset from the UCI Machine Learning Repository.

Files included:

* `winequality-red.csv`
* `winequality-white.csv`

These datasets contain physicochemical measurements of Portuguese "Vinho Verde" wines along with quality scores assigned by human experts.

### Citation

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547–553.

Dataset:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Objectives

The goals of this project are to:

* Develop a reproducible machine learning workflow.
* Explore relationships between physicochemical properties and wine quality.
* Compare multiple regression and classification algorithms.
* Evaluate model performance using appropriate metrics.
* Demonstrate best practices in project organization, version control, and documentation.

## License

The source code in this repository is licensed under the MIT License (if a LICENSE file is included).

The Wine Quality dataset remains the property of its original authors and is used in accordance with the terms provided by the UCI Machine Learning Repository.
