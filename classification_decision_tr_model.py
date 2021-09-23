import os
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.multiclass import OneVsRestClassifier

ps = PorterStemmer()

# Downloading required packages for nltk
nltk.download('stopwords')
nltk.download('punkt')


dataset = pd.DataFrame()
tfidf_vectorizer = TfidfVectorizer()


def data_preprocess(raw_text):
    raw_text = re.sub("\'", "", raw_text)
    raw_text = re.sub("  ", " ", raw_text)
    raw_text = re.sub("[^a-zA-Z]", " ", raw_text)
    raw_text = ' '.join(raw_text.split())
    raw_text = raw_text.lower()
    tokenized_text = word_tokenize(raw_text, language="english")
    no_stopwords_text = [
        x for x in tokenized_text if x not in stopwords.words("english")]
    stemmed_text = [ps.stem(x) for x in no_stopwords_text]
    return " ".join(stemmed_text)


def train_model():
    print('importing data...')
    init_pd = pd.read_csv(
        os.getcwd()+"/datasets/classification_datasets/initial_dataset/plot_summaries.txt", sep="\t", header=None)
    init_pd.columns = ['movie_id', 'plot']

    init_md = pd.read_csv(
        os.getcwd()+"/datasets/classification_datasets/initial_dataset/movie.metadata.tsv", sep="\t", header=None)
    init_md.columns = ['movie_id', 'unique_id', 'movie_name',
                       'released_date', 4, 5, 'language', 'country', 'genre']

    combined_md = pd.merge(
        init_md[['movie_id', 'movie_name', 'genre']], init_pd)

    genres_list = [list(json.loads(str(x).lower()).values())
                   for x in combined_md['genre']]

    new_md = combined_md.copy()
    new_md['genre'] = genres_list

    valid_md = new_md[(new_md['genre'].str.len() != 0)]

    unique_genres = list({x for l in valid_md['genre'].to_list() for x in l})

    print('preprocessing data...')
    new_md['plot'] = new_md['plot'].apply(lambda x: data_preprocess(x))

    print('dumping clean data...')
    # dump
    with open(os.getcwd()+'/datasets/classification_datasets/clean_movie_data_classif.pickle', 'wb') as f:
        pickle.dump(new_md, f)

    # load
    load = pickle.dump
    with open(os.getcwd()+'/datasets/classification_datasets/clean_movie_data_classif.pickle', 'rb') as f:
        load_md = pickle.load(f)

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(load_md['genre'])

    # transform target variable
    y = multilabel_binarizer.transform(load_md['genre'])

    print('saving binarizer...')
    # dump the ml binarizer
    with open(os.getcwd()+'/models/classification_models/dtr_ml_binarizer.pickle', 'wb') as f:
        pickle.dump(new_md, f)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

    # split dataset into training and validation set
    xtrain, xval, ytrain, yval = train_test_split(
        load_md['plot'], y, test_size=0.2, random_state=9)

    # create TF-IDF features
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)

    print('saving tfidfVectorizer...')
    # dump
    with open(os.getcwd()+'/Classif_PlotSum/dtr_tf_idf_vectorizer.pickle', 'wb') as f:
        pickle.dump(new_md, f)

    # Binary Relevance--> onevsrestclassifier

    # Performance metric-->f1

    dtr = DecisionTreeRegressor()
    clf = OneVsRestClassifier(dtr)

    # fit model on train data
    clf.fit(xtrain_tfidf, ytrain)

    print('saving lsvm model...')
    # dump the model
    with open(os.getcwd()+'/models/classification_models/dtr_ovr_classif.pickle', 'wb') as f:
        pickle.dump(new_md, f)


def get_predictions(input):
    # load the model
    load = pickle.dump
    with open(os.getcwd()+'/models/classification_models/dtr_ovr_classif.pickle', 'rb') as f:
        dtr_ovr_classif_model = pickle.load(f)

    clean_input = data_preprocess(input)

    # load the tfidf vectorizer
    with open(os.getcwd()+'/models/classification_models/dtr_tf_idf_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # create TF-IDF features for input
    input_series = pd.Series(clean_input)

    # For testing purposes
    # print("\n\nclean data: ", clean_input, "\nseries: ", input_series, "\n\n")

    # tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    input_tfidf = tfidf_vectorizer.transform(input_series)

    input_prediction = dtr_ovr_classif_model.predict(input_tfidf)

    with open(os.getcwd()+'/models/classification_models/dtr_ml_binarizer.pickle', 'rb') as f:
        multilabel_binarizer = pickle.load(f)

    return multilabel_binarizer.inverse_transform(input_prediction)


if __name__ == '__main__':
    # train_model()
    input = "The Croods is a 2013 American computer-animated adventure comedy film produced by DreamWorks Animation and distributed by 20th Century Fox."
    print(get_predictions(input))
