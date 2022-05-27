# importing
import re
import numpy as np
import pandas as pd
import sys
import pickle
import warnings
from sklearn.metrics import accuracy_score, precision_score, f1_score, make_scorer, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
warnings.simplefilter('ignore')




def load_data(database_filepath):
    print('Strating Loading\n\n\n\n')
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM df", engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    print('complete Loading \n\n\n\n')
    return X,y,list(y.columns.values)


def tokenize(text):
    #print('Strating Tokenizing\n\n\n\n')
    text = re.sub(r"[^a-zA-Z]", " ", text.lower()) 
    tokens = word_tokenize(text)
    stem = [PorterStemmer().stem(word) for word in tokens if word not in stopwords.words("english")]
    #print('complete Tokenizing \n\n\n\n')
    return stem

def calculate_score(y_true, y_pred):
    print('Strating score\n\n\n\n')
    scores = []
    for i in range(np.shape(y_pred)[1]):
        score = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        scores.append(score)
        
    final_score = np.median(scores)
    print('done score \n\n\n\n')
    return final_score


def build_model():
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=False)))
        ]
    )
    parameters = {
        'vect__min_df': [1, 3, 5],
        'tfidf__use_idf':[True, False],
        'clf__estimator__n_estimators':[10, 20, 30], 
        'clf__estimator__min_samples_split':[3, 6, 9]
    }
    scorer = make_scorer(calculate_score)
    grid_CV = GridSearchCV(pipeline, param_grid = parameters,scoring                            
    return grid_CV


def evaluate_model(model, X_test, Y_test, category_names):
    print('Strating eval\n\n\n\n')
    y_pred = model.predict(X_test)
    y_true = np.array(Y_test)

    df = pd.DataFrame()
    for i,target in enumerate(category_names):
        # calculate table
        precision = precision_score(y_true[:, i], y_pred[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        df = df.append({'index':target,'Accuracy':accuracy,'F1 Score':f1,'Precision':precision,'Recall':recall},ignore_index = True)
    print('complete eval \n\n\n\n')

def save_model(model, model_filepath):
    print('Strating saving \n\n\n\n')
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    print('complete saving \n\n\n\n')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print(X_train)
        print(Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()