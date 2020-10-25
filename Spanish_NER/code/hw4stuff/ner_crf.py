from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
from sklearn.metrics import precision_recall_fscore_support
from nltk.tag.stanford import StanfordPOSTagger
from bert_embedding import BertEmbedding 
import string
import scipy.stats
from sklearn.metrics import make_scorer
#from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import spacy

nlp = spacy.load('es_core_news_md')
#bert = BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')
#tagger = StanfordPOSTagger('stanford-models/models/spanish.tagger', 'stanford-models/stanford-postagger.jar')
punctuations = string.punctuation
brown_dict = {}
with open("paths.txt","r") as read:
    for line in read:
        line = line.strip('\n')
        stuff = line.split('\t')
        brown_dict[stuff[1]] = stuff[2]
#print(brown_dict)
#def get_prefix_suffix(word):
    #if
def isPunct(word):
    if(word[0] in punctuations):
        return True
    else:
        return False
def word2features(sent,i):
    #vecs = nlp(sent)
    #print(vecs[0].vector)
    word = sent[i][0]
    #vec = nlp(word)
    #print((nlp(word)).vector)
    #tag = tagger.tag(word)
    #embeddding = 
    tag = sent[i][1]
    # for j in word:
    #     if(j in punctuations):
    #         isPunct = True
    #     else:
    #         isPunct = False

    features = {
        "word": word,
        "isDigit":word.isdigit(),
        "lowercase":word.lower(),
        "isnumber":word.isnumeric(),
        'word.istitle()': word.istitle(),
        "isalphanum":word.isalnum(),
        'length':len(word),
        "isFirstUpper":word[0].isupper(),
        "postag":tag,
        "punctuation":isPunct(word),
        "browncluster":brown_dict[word]
        #"embedding":(nlp(word)).vector
    }
    #print(i)
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            "-1lowercase": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
            "-1:postag": postag1,
            "-1punctuation":isPunct(word1),
            "-1isnumber":word1.isnumeric(),
            "-1browncluster":brown_dict[word1]
            #"embedding":(nlp(word1)).vector
        })
    else:
        features['BOS'] = True
    if i < (len(sent)-1):
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            "+1lowercase": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
            "+1:postag": postag1,
            "+1punctuation":isPunct(word1),
            "+1isnumber":word1.isnumeric(),
            "+1browncluster":brown_dict[word1]
            #"embedding":(nlp(word1)).vector
        })
    else:
        features['EOS'] = True
    if i>1:
        word1 = sent[i-2][0]
        postag1 = sent[i-2][1]
        features.update({
            "-2lowercase": word1.lower(),
            "-2:word.istitle()": word1.istitle(),
            "-2:word.isupper()": word1.isupper(),
            "-2:postag": postag1,
            "-2punctuation":isPunct(word1),
            "-2isnumber":word1.isnumeric(),
            "-2browncluster":brown_dict[word1]
            #"embedding":(nlp(word1)).vector
        })
    if i <(len(sent)-2):
        word1 = sent[i+2][0]
        postag1 = sent[i+2][1]
        features.update({
            "+2lowercase": word1.lower(),
            "+2:word.istitle()": word1.istitle(),
            "+2:word.isupper()": word1.isupper(),
            "+2:postag": postag1,
            "+2punctuation":isPunct(word1),
            "+2isnumber":word1.isnumeric(),
            "+2browncluster":brown_dict[word1]
            #"embedding":(nlp(word1)).vector
        })


    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]


if __name__ == "__main__":

    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb"))

    train_feats = []
    train_feats = [sent2features(s) for s in train_sents]
    train_labels = []
    train_labels = [sent2labels(s) for s in train_sents]
    # for sent in train_sents:
    #     for i in range(len(sent)):
    #         #print(i)
    #         #train_labels.append(sent[i][-1])
    #         feats = word2features(sent,i)
    #         train_feats.append(feats)
    #print(train_sents[2])
    #print(train_feats[2])
   # print(set(train_labels))

    model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.1,c2=0.1,max_iterations=100,all_possible_transitions=True)
    #vectorizer = DictVectorizer()
    #train_feats = vectorizer.fit_transform(train_feats)
    model.fit(train_feats, train_labels)

    test_feats = [sent2features(s) for s in test_sents]
    test_labels = [sent2labels(s) for s in test_sents]

    # for sent in dev_sents:
    #     for i in range(len(sent)):
    #         test_labels.append(sent[i][-1])
    #         feats = word2features(sent,i)
    #         test_feats.append(feats)

    y_pred = model.predict(test_feats)
    labels = list(model.classes_)
    labels.remove('O')
    print(metrics.flat_f1_score(test_labels, y_pred,average='weighted', labels=labels))
    j = 0
    #print(y_pred[0])
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
         for sent in test_sents:
            for i in range(len(sent)):
                 word = sent[i][0]
                 gold = sent[i][-1]
                 pred = y_pred[j][i]
                 out.write("{}\t{}\t{}\n".format(word, gold, pred))
            j+=1
         out.write("\n")

    print("Now run: python conlleval.py results.txt")
