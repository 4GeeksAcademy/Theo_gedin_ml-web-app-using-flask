# ML Web App Using Flask

A machine learning web application built with Flask for educational purposes.

## Project Instructions

### Step 1: Find a Dataset
Research different online sources about different datasets that you could use to train a model. You can use some public API, the UCI repository for Machine Learning or the Kaggle section of datasets, among many other sources. Remember to look for a simple dataset as this is not the final project of the course.

**Recommended Data Sources:**
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Scikit-learn Built-in Datasets](https://scikit-learn.org/stable/datasets.html)
- [Google Dataset Search](https://datasetsearch.research.google.com/)

### Step 2: Develop a Model
Once you have found your ideal data set, analyze it and train a model. Optimize it if necessary.

**Tasks to complete:**
- Perform exploratory data analysis (EDA)
- Clean and preprocess the data
- Select appropriate features
- Train a machine learning model
- Evaluate model performance
- Optimize hyperparameters if needed
- Save the trained model for deployment

### Step 3: Develop a Simple Web Application Using Flask
Create a simple Flask web application with Bootstrap styling for a clean, responsive interface.

**Requirements:**
- Simple Flask app with prediction form
- Bootstrap for responsive styling  
- Model prediction endpoint
- Basic error handling
- Clean, educational-focused design

### Step 4: Deploy to Render
Deploy the simple Flask application to Render for online access.

**Deployment steps:**
- Test the app locally first
- Create requirements.txt with dependencies
- Deploy to Render
- Test the deployed application

## Project Structure
```
├── app.py                      # Main Flask application
├── models/                     # Saved ML models
│   ├── mathe_model.pkl         # Trained Random Forest model
│   └── encoders.pkl            # Label encoders for categorical data
├── templates/                  # HTML templates (Bootstrap styled)
│   ├── index.html              # Home page with prediction form
│   └── result.html             # Prediction results page
├── notebooks/                  # Jupyter notebook for model training
│   └── mathe_model_training.ipynb
├── MathE dataset.csv           # Training dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage
1. Open the web application
2. Fill in the student information:
   - Select student's country
   - Choose question level (Basic/Advanced)  
   - Select mathematics topic
   - Select subtopic
3. Click "Predict" to see if the student will answer correctly
4. View the prediction result and confidence score

## Project Overview

**Dataset:** [MathE Dataset for Assessing Mathematics Learning in Higher Education](https://archive.ics.uci.edu/dataset/1031/dataset+for+assessing+mathematics+learning+in+higher+education)

**What the App Does:**
This Flask web application predicts whether a student will answer a mathematics question correctly based on:

**Input:**
- Student's Country (Portugal, Italy, Ireland, Lithuania, etc.)
- Question Level (Basic/Advanced)
- Topic (Statistics, Differentiation, Linear Algebra, etc.)
- Subtopic (specific mathematical concept)

**Output:**
- Prediction: Correct (1) or Incorrect (0) answer
- Confidence score (probability)

**Model:** Random Forest classifier trained on 9,500+ student responses from European universities, achieving educational insights about student performance patterns across different countries and mathematical topics.

## Model Information
- **Algorithm:** Random Forest Classifier
- **Features:** Country, Difficulty Level, Topic, Subtopic
- **Target:** Type of Answer (0=Incorrect, 1=Correct)
- **Training Data:** 9,548 student responses from European mathematics courses

## Deployment
https://math-performance-predictor.onrender.com

## External Resources Used
- **Bootstrap 5**: For responsive UI styling
- **Flask**: Python web framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **Pickle**: Model serialization

## License
This project is for educational purposes.
