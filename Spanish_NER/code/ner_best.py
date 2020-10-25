from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import sklearn_crfsuite
from nltk.tag.stanford import StanfordPOSTagger 
import string
#import tensorflow_hub as hub
from bert_embedding import BertEmbedding

bert = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
#print(bert('espanol'))
#sentence = ("autralia espanol").split(' ')
#embed = bert(sentence)
#first_word = embed[0]
#print(first_word[1])
# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.

tagger = StanfordPOSTagger('stanford-models/models/spanish.tagger', 'stanford-models/stanford-postagger.jar')
punctuations = string.punctuation

def get_bert_embeddings(sent):
    print('yep')

def get_pos_tags(sent):
    return 0

def getfeats(word, o,tag):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    #tagger = conll2002.tagged_words()
    #print(spanish_postagger.tag(word))
    #print('the tag for '+word+' is '+tag)
    o = str(o)
    lower = word.lower()
    #punct = word.punctuation()
    for i in word:
        if(i in punctuations):
            isPunct = True
        else:
            isPunct = False
    #print(isPunct)
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
            (o + "isnumber",word.isnumeric()),
            (o + "isalphanum",word.isalnum()),
            (o + 'length',len(word)),
            (o + "isFirstUpper",word[0].isupper()),
            (o + "postag",tag),
            (o + "contains punctuation",isPunct)
        ]
    else:
        features = [
            (o + "word", word),
            # TODO: add more features here.
            (o + "lowercase",word.lower()),
            (o + "isnumber",word.isnumeric()),
            (o + "isalphanum",word.isalnum()),
            (o + 'length',len(word)),
            (o + "isFirstUpper",word[0].isupper()),
            (o + 'prefix',prefix),
            (o + 'suffix',suffix),
            (o + "postag",tag),
            (o + "contains punctuation",isPunct)
        ]
    #print(features)
    return features


def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    #constructing the sentece
    # s = []
    # for j in range(len(sent)):
    #     s.append(sent[j][0])
    # embeds = bert(s)
    #print('The sentence is:',s)
    #tag = spanish_postagger.tag(sent)
    #print(tag,word)
    # the window around the token
    for o in [-2,-1,0,1,2]:
        if i + o >= 0 and i + o < len(sent):
            word = sent[i + o][0]
            tag = sent[i+o][1]
            #tag = tagger.tag(s[i])
            #print('embedding',embeds[i])
            #print('pos tag',tag)
            featlist = getfeats(word, o,tag)
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
        #print(sent)
        #s = sent.split(' ')
        #print(s)
        # s = []
        # #pos_tags = []
        # for j in range(len(sent)):
        #     s.append(sent[j][0])
        # pos_tags = tagger.tag(s)
        # #print(s)
        # print(pos_tags)
        #     #pos_tags.append()
        # embeds = bert(s)
        # print(len(embeds[0][1]))
        # print(embeds[0][1][0].shape)
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])
    #print(train_feats)
    #vectorizer = DictVectorizer()
    #X_train = vectorizer.fit_transform(train_feats)
    #X_train = train_feat
    X_train = train_feats
    print(X_train[0])
    print(train_labels[0])
    # TODO: play with other models
    model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=True)
    #crf.fit(X_train, y_train)
    #model = SGDClassifier()
    #model = PassiveAggressiveClassifier()
    #model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=True )
    #print(X_train[0],train_labels[0])
    model.fit(X_train, train_labels)

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
