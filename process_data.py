# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:35:43 2020

@author: hto_r
"""

# import packages
import sys
import pandas as pd
from sqlalchemy import create_engine

def run_process_data(data_path_messages, data_path_categories):
    """
    Reads, cleans and transform two .csv files with the messages and the labelling
    Returns 1 dataframe with the cleaned data 
    """
    #read in the .csv data files
    messages = pd.read_csv(data_path_messages)
    categories = pd.read_csv(data_path_categories)
    #Mmerge both data frames
    df = messages.merge(categories)
    #makes a new df with expanded categories
    categories = df.categories.str.split(pat=";", expand=True)
    #splits the categories column into separate, clearly named colums
    row = categories.iloc[0].tolist()
    category_colnames = [x.split("-")[0] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split(pat="-", expand=True)[1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df=df[~df.duplicated()]
    #drop the rows with a value of 2 in 'related' in order to obtain binary data
    df=df[~pd.Series(df.related==2)]
    return (df)

def export_data(df, database_path):
    """
    Exports a dataframe into a table names 'messages_cleaned' of a sql lite database 
    """
    engine = create_engine('sqlite:///'+database_path)
    df.to_sql('messages_cleaned', engine, index=False)


if __name__ == '__main__':
    """
    The script takes the file paths of the two datasets and database, 
    cleans the datasets, and stores the clean data into a SQLite database 
    in the specified database file path.
    """
    data_path_messages = sys.argv[1]# get filename of data file 1
    data_path_categories = sys.argv[2]# get filename of data file 2
    database_path = sys.argv[3]# get filename of database to be stored
    df=run_process_data(data_path_messages, data_path_categories)  # run data pipeline
    export_data(df, database_path)