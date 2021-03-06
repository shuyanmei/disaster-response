# Disaster Response Pipeline Project
In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files
1. ETL Pipeline

data/process_data.py is a data cleaning pipeline that loads the messages and categories datasets, merges them together and store it into a SQLite database after cleaning.

2. ML Pipeline

model/train_classifier.py is a machine learning pipeline that load, split the data and train, tune, evaluate and write the model.(There is a 100mb limit, so the pickle file is not uploaded here)

3. Web App

app/run.py is the flask code for the web app and data visualizations.
app/template/html are the html files
