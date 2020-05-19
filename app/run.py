import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_cleaned', engine)

# load model
model = joblib.load("../models/classifier.pkl")

def make_histogram_lists(l, bins=20):
    np_list=np.array(l)
    count_list=[]
    bin_right_edge=[]
    bin_min=0#int(np_list.mean()-3*np_list.std())
    bin_max=int(np_list.mean()+3*np_list.std())
    for current_bin in range(bin_min, bin_max, int((bin_max-bin_min)/bins)):
        bin_right_edge.append(current_bin)
        count_list.append(len(np_list[np_list<current_bin]))
        np_list=np_list[np_list>=current_bin]
    return count_list, bin_right_edge

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    caracter_count=list(df['message'].apply(lambda x: len(x)))
    hist_caracter_y, hist_caracter_x=make_histogram_lists(caracter_count, bins=20)
    
    df['tokens']=df['message'].apply(lambda x: len(tokenize(x)))
    token_count=list(df['tokens'])
    hist_tokens_y, hist_tokens_x=make_histogram_lists(token_count, bins=20)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
               Bar(
                   x=hist_caracter_x,
                   y=hist_caracter_y
                )
            ],

            'layout': {
                'title': 'How many Caracters do the messages have?',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Caracters"
                }
            }
        },
        
       {
            'data': [
                Bar(
                    x=hist_tokens_x,
                    y=hist_tokens_y
                )
            ],

            'layout': {
                'title': 'Distribution of tokens of the messages?',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        
      {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'What is the Genre of the messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()