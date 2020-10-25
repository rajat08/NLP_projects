from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.metrics import precision_score

#import our_metrics

TRAIN_FILE = Path("raw_data/GunViolence/train.tsv")
DEV_FILE = Path("raw_data/GunViolence/dev.tsv")
TEST_FILE = Path("raw_data/GunViolence/test.tsv")

LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# These frames/labels correspond to
# 1) Gun or 2nd Amendment rights
# 2) Gun control/regulation
# 3) Politics
# 4) Mental health
# 5) School or public space safety
# 6) Race/ethnicity
# 7) Public opinion
# 8) Society/culture
# 9) Economic consequences

# function to remove stopwords
def remove_stopwords(rev):
  stop_words = stopwords.words('english') 
  rev_new = " ".join([i for i in rev if i not in stop_words])
  return rev_new


def load_data_file(data_file):
    """Load newsframing data

    Returns
    -------
    tuple
        First element is a list of strings(headlines)
        If `data_file` has labels, the second element
        will be a list of labels for each headline.
        Otherwise, the second element will be None.
    """
    print("Loading from {} ...".format(data_file.name), end="")
    text_col = "news_title"
    theme1_col = "Q3 Theme1"

    with open(data_file) as f:
        df = pd.read_csv(f, sep="\t")
        X = df[text_col].tolist()



        y = None
        if theme1_col in df.columns:
            y = df[theme1_col].tolist()

        print(
            "loaded {} lines {} labels ... done".format(
                len(X), "with" if y is not None else "without"
            )
        )
    items = []
    
    nlp = spacy.load('en_core_web_sm')
    #using stop words to remove
    stop_words = stopwords.words('english') 
    for item in stop_words:
        nlp.vocab[item].is_stop = True
            
    for item in X:
        sentence = []
        doc = nlp(item)
        for token in doc:
            if not token.is_punct:
                x = token.lemma_
                if x not in stop_words:
                    sentence.append(x)
        items.append(sentence)
        
    print(items[0])  
    
    new_list = []
    for item in items:
        sentence = ""
        for x in item:
            sentence = sentence + " " + x
        new_list.append(sentence)
            
            
    return (new_list, y)


def build_naive_bayes():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    nb_pipeline = None
    estimator = []
    estimator.append(("CountVectorizer", CountVectorizer()))
    estimator.append(("NaiveBayes", ComplementNB()))
    nb_pipeline = Pipeline(estimator)
    return nb_pipeline


def build_logistic_regr():
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    logistic_pipeline = None
    logistic = LogisticRegression()
    estimator = []
    estimator.append(("CountVectorizer", CountVectorizer()))
    estimator.append(("Logistic",logistic))
    logistic_pipeline = Pipeline(estimator)

    return logistic_pipeline


def build_own_pipeline() -> Pipeline:
    """

    Returns
    -------
        Pipeline
        An sklearn Pipeline
    """
    ##### Write code here #######
    pipeline = None
    SVM = SVC(C=4, kernel="linear")
    estimator = []
    estimator.append(("CountVectorizer", CountVectorizer()))
    estimator.append(("SVC", SVM))
    pipeline = Pipeline(estimator)
    ##### End of your work ######
    return pipeline


def output_predictions(pipeline):
    """Load test data, predict using given pipeline, and write predictions to file.

    The output must be named "predictions.tsv" and must have the following format.
    Here, the first three examples were predicted to be 7,2,3, and the last were
    predicted to be 6,6, and 2.

    Be sure not to permute the order of the examples in the test file.

        7
        2
        3
        .
        .
        .
        6
        6
        2

    """
    ##### Write code here #######
    model = pipeline
    X_test,y_test = load_data_file(TEST_FILE)
    y_pred = model.predict(X_test)
    print("y-test------" + str(y_test))
    print("y-pred-------" + str(y_pred))
    prediction = pd.DataFrame(y_pred).to_csv('predictions.tsv', sep='\t', index = None) 
    ##### End of your work ######


def preprocess(article):
    print('spaCy Version: %s' % (spacy.__version__))
    spacy_nlp = spacy.load('en_core_web_sm')
    print('Number of stop words: %d' % len(spacy_stopwords))
    print('First ten stop words: %s' % list(spacy_stopwords)[:10])
    doc = spacy_nlp(article)
    ndoc = []
    for token in doc:
        if not token.is_punct:
            ndoc.append(token)
    return ndoc


def main():
    #Read data
    X_train, y_train_true = load_data_file(TRAIN_FILE)
    X_dev, y_dev_true = load_data_file(DEV_FILE)
    svm_pipeline = build_own_pipeline()
    svm_pipeline.fit(X_train,y_train_true)
    preds_dev_svm = svm_pipeline.predict(X_dev)
    accuracy_svm = accuracy_score(y_dev_true, preds_dev_svm)
    print("SVM accuracy : " + str(accuracy_svm*100))
    output_predictions(svm_pipeline)


if __name__== "__main__":
    main()