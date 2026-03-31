# SMS-Spam-Detection
This project builds a spam detection system using NLP and machine learning. We preprocess text, extract features, apply TF-IDF, and train multiple classifiers to identify spam messages accurately. 
# Overview:
This project is a Data Analytics project using Python, focused on analyzing and understanding SMS messages to classify them as Spam or Ham (Not Spam). It combines data analysis, visualization, and basic machine learning techniques to extract insights and build a predictive model.
# Objective:
Perform data cleaning and preprocessing,
Conduct Exploratory Data Analysis (EDA),
Visualize patterns in spam and ham messages,
Extract meaningful features from text data,
Build and evaluate basic machine learning models,
# Dataset:
The dataset contains SMS messages labeled as:
Spam (1) → Unwanted messages,
Ham (0) → Normal messages
# Technologies Used:
Python ,
Pandas & NumPy (Data Analysis),
Matplotlib & Seaborn (Visualization),
NLTK (Text Processing),
Scikit-learn (Machine Learning).
# Data Cleaning:
Removed unnecessary columns,
Renamed columns for better understanding,
Handled missing values,
Converted text data into proper format.
# Exploratory Data Analysis (EDA):
Analyzed distribution of spam vs ham messages
Created new features:
Number of characters,
Number of words,
Number of sentences,
Used visualizations:
Count plots,
Histograms,
Boxplots,
Heatmaps,
WordCloud.
# Text Processing:
Converted text to lowercase,
Tokenization,
Removed stopwords and punctuation,
Applied stemming.
# Key Insights:
Spam messages tend to have more characters and words,
Certain words appear frequently in spam messages,
Data is imbalanced (more ham than spam),
Text preprocessing improves analysis quality.
# Machine Learning (Basic):
Applied TF-IDF vectorization,
Trained models like:
Naive Bayes,
Logistic Regression,
SVM,
Compared models using accuracy and precision.
# Results:
Naive Bayes performed well for text classification,
The model can effectively distinguish spam messages.
