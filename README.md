# Disaster Response Pipeline

## Description
This project is part of the Udacity Data Scientist for Enterprise project. The dataset contains labelled emergency messages from real-life disasters. The aim of the project is to train a model to classify the labels of the messages.

In this project, an ETL pipeline is built that cleans the data using regex and NLTK, transforms it, then saves it to a database. The data is then loaded to train a multi-output classifier model with a random forest as an estimator. The model is then saved and used in a Flask application that predicts the categories of a given input.

## Getting Started

### Pre-requesites

* NumPy
* Sklearn
* NLTK
* Pandas
* Flask
* SQLAlchemy
* Plotly

### Files

* **run.py**: Flask app with data visualisation and predictions.
* **process_data.py**: Code that takes the address of a category file and the texts, transforms target variables, and saves the data in a database.
* **train_classifier.py**: Script to take the data from the database, build a pipeline, and performs Grid Search cross-validation. The best estimator is saved as a pickel file.

### Cloning

To run the code, clone this GIT repository:

`git clode https://github.com/SaadChaouki/disaster-response-project.git`

### Executing

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Web Application

![alt text](https://raw.githubusercontent.com/SaadChaouki/disaster-response-project/screenshots/overall_app.png)

### Author

* [Saad Chaouki](https://www.linkedin.com/in/schaouki/)

### Acknowledgements

* [Udacity](https://www.udacity.com/)
