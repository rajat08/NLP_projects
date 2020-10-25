from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import sklearn_crfsuite
from hmmlearn import hmm
# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.
#def is_abbreviation(abbrev):
#    abbrev = abbrev.lower()


def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    lower = word.lower()
    if(len(lower)>4):
        prefix = lower[0:4]
        suffix = lower[-4:]
    else:
        prefix = ''
        suffix = ''
    if(o!=0):
        features = [
            (o + "word", word),
            (o + "lowercase",word.lower()),
            (o + "is a number",word.isnumeric()),
            (o + 'length',len(word)),
            (o + "isFirstUpper",word[0].isupper()),
        ]
    else:
        features = [
            (o + "word", word),
            # TODO: add more features here.
            (o + "lowercase",word.lower()),
            (o + "is a number",word.isnumeric()),
            (o + 'length',len(word)),
            (o + "isFirstUpper",word[0].isupper()),
            (o + 'prefix',prefix),
            (o + 'suffix',suffix)
        ]
    #print(features)
    return features


def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    for o in [-2,-1,0,1,2]:
        if i + o >= 0 and i + o < len(sent):
            word = sent[i + o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)

    return dict(features)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb"))

    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    #model = Perceptron()
    model = hmm.GaussianHMM(n_components=3)
    #model.startprob_ = np.array([0.6, 0.3, 0.1])
    model = hmm.GaussianHMM(n_components=3, n_iter=100)
    
    X_train = X_train[:100]
    X_train  = X_train.todense()
    #model.fit(X_train)
    #model = SGDClassifier()
    #model = PassiveAggressiveClassifier()
    #model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=True )
    model.fit(X_train, train_labels[:100])

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
