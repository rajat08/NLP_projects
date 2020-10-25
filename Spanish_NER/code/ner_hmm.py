from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
import sklearn_crfsuite
from sklearn.metrics import precision_recall_fscore_support
from nltk.tag.stanford import StanfordPOSTagger
from bert_embedding import BertEmbedding 
import string
import scipy.stats
from collections import Counter,defaultdict
from sklearn.metrics import make_scorer
#from hmmlearn import hmm
from hmmlearn.hmm import MultinomialHMM
#from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import spacy
import numpy as np


if __name__ == "__main__":

    # Load the training data
    train_sents = list(conll2002.iob_sents("esp.train"))
    dev_sents = list(conll2002.iob_sents("esp.testa"))
    test_sents = list(conll2002.iob_sents("esp.testb"))

    n_states = 9
    start_prob = []
    start_count = {}
    states = np.array(['I-PER', 'B-LOC', 'B-ORG', 'I-ORG', 'I-LOC', 'B-PER', 'O', 'I-MISC', 'B-MISC'])
    #print(states)
    for state in states:
        start_count[state] = 0
    #{'I-PER', 'B-LOC', 'B-ORG', 'I-ORG', 'I-LOC', 'B-PER', 'O', 'I-MISC', 'B-MISC'}
    
    
    trans_mat = np.zeros((9,9)) 
    emission_dict = defaultdict(list)
        
    
    for sent in train_sents:
        for i in range(len(sent)):
            word = sent[i][0]
            state = sent[i][-1]
            ind1 = np.where(states==state)
            ind1 = ind1[0][0]
            if(i==0):
                start_count[state] += 1
            if(i<(len(sent)-1)):
                n_state = sent[i+1][-1]
                ind2 = np.where(states==n_state)
                ind2 = ind2[0][0]
                trans_mat[ind1,ind2] += 1
            if(word not in emission_dict):
                emission_dict[word] = [0]*9
                tmp = emission_dict[word]
                tmp[ind1] += 1
                emission_dict[word] =tmp
                
            else:
                tmp = emission_dict[word]
                tmp[ind1] += 1
                emission_dict[word] =tmp

    trans_mat /= trans_mat.sum(axis=1).reshape(-1, 1)
    #print(sum(trans_mat[0]))

    start_values = list(start_count.values())
    #print(start_values[0])
    
    for i in range(9):
        start_prob.append(start_values[i]/len(train_sents))
    
    emission_matrix = []
    for key in emission_dict.keys():
        tmp = emission_dict[key]
        emission_matrix.append(tmp)
    
    emission_matrix = np.array(emission_matrix)
    
    #Adding one row for unknowns
    unk = np.zeros((1,9))
    emission_matrix = np.vstack((emission_matrix,unk))
    emission_matrix = emission_matrix.T

    model = MultinomialHMM(n_components=n_states,algorithm='viterbi')
    model.startprob_= np.array(start_prob)
    model.transmat_ = trans_mat
    model.emissionprob_ = emission_matrix
    
    # format is: word gold pred
    nexcept = 0
    with open("results_hmm.txt", "w") as out:
         for sent in test_sents:
            inp = []
            for i in range(len(sent)):
                word = sent[i][0]
                try:
                    k = list(emission_dict.keys()).index(word) 
                except:
                    nexcept+=1
                    k = emission_matrix.shape[0]-1
        
                inp.append(k)
               
            inp = np.atleast_2d(inp).T
           
            if(len(inp)!=1): 
                logprob,out_ = model.decode(inp,algorithm='viterbi')
                out_ = list(map(lambda x: states[x], out_))
               
            else:
                #print(sent[0][-1])
                out_ =[]
                out_.append('O')
    
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = out_[i]
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
            #j+=1
         out.write("\n")
    #print(nexcept)
    print("Now run: python conlleval.py results_hmm.txt")
