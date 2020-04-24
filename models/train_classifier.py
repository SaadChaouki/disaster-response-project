# import libraries
import pandas as pd
import numpy as np
import pickle
from joblib import dump, load
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import re
import sys

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer 

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Download nltk ressources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('select * from emergency_messages', engine)
    X = df['message']
    Y = df.drop(['message', 'genre'], axis = 1)
    return X, Y, Y.columns

def wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize(text):
    contract_dict = {"ain't": "am not", "aren't": "am not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would",  "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
    "how's": "how is", "i'd": "I would", "i'd've": "I would have", "i'll": "I will", "i'll've": "I will have",
    "i'm": "i am", "i've": "I have", "isn't": "is not", "it'd": "it would","it'd've": "it would have",
    "it'll": "t will", "it'll've": "it will have", "it's": "t is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have","will've": "will have", "won't": "will not","won't've": "will not have",
    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
    "you're": "you are", "you've": "you have"}

    # Finding and replacing urls
    text = re.sub('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',  'placeholder', text)

    # Replace contractions
    for key in contract_dict: 
        text = text.replace(key.lower(), contract_dict[key])
    
    # Remove special characters
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    
    # Lower
    text = text.lower()
    
    # stop words
    stop_words = stopwords.words("english")
    
    # Tokens
    tokens = word_tokenize(text)
    
    # Part of speech
    pos_tokens = pos_tag(tokens)
    
    # Lemmatize
    clean_tokens = [WordNetLemmatizer().lemmatize(token[0], wordnet_pos(token[1])) for token in pos_tokens if token[0] not in stop_words]
    
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': [(1, 1), (1,2)],
         'vect__max_df': [0.75, 1.0],
         'tfidf__use_idf': [True, False],
        'model__estimator__n_estimators': [50, 100]
    }

    # Grid search
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 2, n_jobs = -1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test), columns = category_names)
    # classification report
    fscores = []
    recalls = []
    precisions = []
    accuracies = []
    
    # Transforming preds to dataframe
    
    y_pred = pd.DataFrame(y_pred, columns = Y_test.columns)
    # Report
    for column in y_pred.columns:
        report = classification_report(Y_test[column],y_pred[column])
#         print(report)
        precision,recall,fscore,_=score(Y_test[column],y_pred[column], average = 'macro')
        accuracies.append(accuracy_score(Y_test[column], y_pred[column]))
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)


    # Printing metrics
    print(f'Overall Accuracy: {round(np.mean(accuracies),2)}')
    print(f'Overall f-score: {round(np.mean(fscores),2)}')
    print(f'Overall recall: {round(np.mean(recalls),2)}')
    print(f'Overall precision: {round(np.mean(precisions),2)}')


def save_model(model, model_filepath):
    file_model = open(model_filepath, 'wb')
    pickle.dump(model, file_model)
    file_model.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
