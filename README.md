# Disaster Response Project
This project creates a classification model for sms messages based on structured and labeled messages made available by udacity.
The interface with the user is an app that in the master page presents the typical length of the messages and their type.
In a query page, a new message can be inputed and a classification is given using the model previousl trained
The model classifies a new message into one or thirty three categories

# Instalation
Run the flask app by running the run.py. Modifiy the this script is necessary to run it on your local server 

# Running the webapp
To start the webapp type python 'run.py' app's directory. The app will be running on your local server. Be sure to note the port you are using to call the app in your browser.

Once on the app you can type in the message and resulting categories will be highlited. 

# Overview of the results 

A straight forward pipeline was chosen with a personilized tokenizer function, and a Tfidf Transformer. As for the model Adaboost dellivered the best results after exploring also Random trees, bagging whilst tuning their respective parameters. The accuracy on the testing set was above 90% for most of the 36 categories. 

# References
To read more about adaboost I recomend the introduction to statistical learning from james gareth
https://faculty.marshall.usc.edu/gareth-james/ISL/

# License
MIT License. The data was made available by Udacity. I do not own this data. 

# GitHUB Link
https://github.com/htorodriguez/disaster_response_project

