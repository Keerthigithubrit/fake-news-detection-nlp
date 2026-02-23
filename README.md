# Fake News Detection-NLP 

## Description
The project ocus on fake news detection using techniques like Machine Learning(ML),natural Language Processing(NLP).The goal is to build a reliable text classification system that can distinguish between fake and real news articles

## Project Overview
The goal is to build a reliable text classification system that can distinguish between fake and real news articles,

2. Data Understanding
2. Data Cleaning& Preprocessing
3. Feature Extraction using TF-IDF
4. Machine Learning & Model Training Process
5. Model Evaluatio (Accuracy,Confusion Matrix, Classification Report)

## Installation
Install required libraries

* Numpy
* Pandas
* Scikit learn
* Matplotlib
* Seaborn
* nltk

```
pip install numpy pandas scikit-learn matplotlib seaborn nltk
```
Download **Stopwords** using nltk.

```
import nltk
nltk.download('stopwords')
```
## Data Understanding
The dataset collected from[Kaggle]((https://www.kaggle.com/)).Load the data using pandas and view the Steps performed:

1. Load dataset using Pandas
2. Understand data structure
3. Check shape and format
4. Identify missing and null values
5. Analyze class distribution

## Data Cleaning
### Binary Encoding

The target labels were encoded as:
* 0 - Fake
* 1 - Real

## Data Preprocessing 
### Text Cleaning
Text cleaning was performed using Regular Expressions and NLP techniques:

1. Convert text to lowercase
1. Remove brackets
2. Remove puntuation
3. Remove non-alphanumeric characters
4. Remove numbers
5. Remove URL/links
3. Remove Stopwords(is,a,the...)
4. Remove extra spaces

This process ensures clean and structured text data for feature extraction.

## Feature Extraction (Vectorization)
Machines understand only numbers, not text. Therefore, text data was converted into numerical format using TF-IDF Vectorization.

* Training data → fit_transform()
* Test data → transform()

```python
# Convert text to number using vectorization
vectorizer = TfidfVectorizer(max_df = 0.7,min_df=5,stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

## Machine Learning 

###  Model Selection
The model used in this project is: *Logistic Regression*

### Reason:
* Works well for binary classification
* Efficient for large text datasets
* Provides good interpretability

## Model Evaluation

```
Accuracy : 1.0

Classification Report:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00       506
           1       1.00      1.00      1.00       494

    accuracy                           1.00      1000
   macro avg       1.00      1.00      1.00      1000
weighted avg       1.00      1.00      1.00      1000

Confusion Matrix:
     [[506   0]
     [  0 494]]
```
![Classification Report plot](images/CR.PNG)

![Confusion Matrix](images/CM.PNG)

### Results
* The model achieved 100% accuracy on the test dataset.
* Precision, Recall, and F1-score are all 1.00.
* The confusion matrix shows zero misclassifications.

## Conclusion:
**Based on the evaluation metrics:**
>The model successfully learned patterns from the dataset.Predictions are highly accurate.Logistic Regression performed effectively for this NLP classification task.This project demonstrates the power of combining NLP preprocessing techniques with Machine Learning models for text classification problems.