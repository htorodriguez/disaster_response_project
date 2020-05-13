# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:35:43 2020

@author: hto_r
"""

# import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pickle


def load_data(database_path):
    """
    Loads a table 'messages_cleaned' for the given sqllite database 
    Returns X and Y vectors of input and outputs and the category names
    """
    
    # read in file
    engine = create_engine('sqlite:///'+database_path)
    df = pd.read_sql("SELECT * FROM messages_cleaned", engine)
    X =df.message 
    Y =df.iloc[:,4:]
    category_names=Y.columns.values
    Y=np.array([Y.iloc[i,:].tolist() for i in range(Y.shape[0])])
    
    return X, Y, category_names



def tokenize(text):
    """
    Custom tokenize function using nltk to case normalize, lemmatize, and tokenize text.
    """    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    words = word_tokenize(text)
    tokens= [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Vectorize and then applies TF-IDF to the tokenized text
    builds a pipeline that processes text 
    performs multi-output classification on the 36 categories in the dataset. 
    GridSearchCV is used to find the best parameters for the model.
    """
    # 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    # define parameters for GridSearchCV
    parameters =  {'clf__estimator__n_estimators': [20, 50], 
                   'clf__estimator__learning_rate':[0.1]}
    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, parameters)

    return model_pipeline

def predict_and_evaluate(model, X_test, Y_test, category_names):
    """
    Predicts a Y vector based on a model and a X_Test dataset
    The f1 score, precision and recall for the test set is outputted 
    for each category.
    """
    
    Y_pred=model.predict(X_test)
    Y_pred_df=pd.DataFrame(Y_pred, columns=category_names)
    Y_test_df=pd.DataFrame(Y_test, columns=category_names)

    for col in category_names:
        y_pred=Y_pred_df[col]
        y_test=Y_test_df[col]
        print('Classification report of: '+ col+'\n')
        print(classification_report(y_test, y_pred))


def train(X, Y, model, category_names):
    """
    Splits the dataset into test and train set
    Trains the model only on the train set
    Uses the test set to perdict and evaluate
    """
    
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # fit model
    model.fit(X_train, Y_train)

    # output model test results

    predict_and_evaluate(model, X_test, Y_test,category_names)
    return model


def export_model(model, model_path):
    """
    Export model as a pickle file to the model_path
    """
    pickle.dump(model, open(model_path, 'wb'))


def run_pipeline(database_path, model_path):
    """
    Reads the data, builds the model, trains the model and exports
    """
    
    X, Y, category_names = load_data(database_path)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, Y, model, category_names)  # train model pipeline
    export_model(model, model_path)  # save model


if __name__ == '__main__':
    """
    The script takes the database file path and model file path, creates and 
    trains a classifier, and stores the classifier into a pickle file to the 
    specified model file path.
    """
    database_path = sys.argv[1]# get filename of database
    model_path = sys.argv[2]# get filename of database
    run_pipeline(database_path, model_path)  # run data pipeline