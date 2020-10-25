import numpy as np
import os 
import vec_spaces
from preprocessing import read_processed_data, DataLoader
from gensim.models import Word2Vec
from numpy import savetxt

def test_newsgroup_similarity(
    mat, id_to_newsgroups, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, D)
            Each column is "newsgroup" vector. 

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############
  
    f = open('ngroup_sim_with_idf.text','a')
    for i in range(mat.shape[0]):
        vals = sim_func(mat[i],mat)
        f.write('similarity for '+id_to_newsgroups[i]+'\n')
        f.write('-------------------------------------------------'+'\n')
        for j in range(len(vals)):
            f.write(id_to_newsgroups[j]+' : '+str(vals[j])+'\n')
        f.write('************************************************'+'\n')
    f.close()
    ####### End of your code #############


def test_word_similarity(
    mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)
            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """

    ####### Your code here ###############
  
    
    f = open('word_sim_with_ppmi.text','a')

    for i in range(mat.shape[0]):
        vals = sim_func(mat[i],mat)
        f.write('similarity for '+id_to_tokens[i]+'\n')
        f.write('-------------------------------------------------'+'\n')
        for j in range(len(vals)):
            f.write(id_to_tokens[j]+' : '+str(vals[j])+'\n')
        f.write('************************************************'+'\n')
    f.close()
    ###### End of your code #############


def test_word2vec_similarity(
    mat, id_to_tokens, sim_func=vec_spaces.compute_cosine_similarity
):
    """

    Arguments
    ---------
        mat: `np.ndarray` (V, d)
            Each row is a d-dimensional word vector.

        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        sim_func: `function(array(M,), array(N,M))->array(N,)`
            A function that returns the similarity of its first arg to every row
            in it's second arg.
    """
    ####### Your code here ###############
    f = open('word2vec_similarity.text','a')
   
    for i in range(20):
        vals = sim_func(mat[i],mat)
        f.write('similarity for '+id_to_tokens[i]+'\n')
        f.write('-------------------------------------------------'+'\n')
        for j in range(len(vals)):
            f.write(id_to_tokens[j]+' : '+str(vals[j])+'\n')
        f.write('************************************************'+'\n')
    f.close()
    ####### End of your code #############


def main():
    ####### Your code here ###############
    data = read_processed_data()
    token_ids = data['id_to_tokens']
    ngroup_ids = data['id_to_newsgroups']
    ngroup_tokens_post = data['newsgroup_and_token_ids_per_post']
    
    
    mat = vec_spaces.create_term_newsgroup_matrix(ngroup_tokens_post,token_ids,ngroup_ids,tf_idf_weighing=True)
    test_newsgroup_similarity(mat,token_ids)
  
    #word content matrix
    #ngroup_tokens_post = np.array(ngroup_tokens_post)
    #print(ngroup_tokens_post.shape)
    w_mat = vec_spaces.create_term_context_matrix(ngroup_tokens_post,token_ids,ngroup_ids,ppmi_weighing=True)
    
    savetxt('data.csv', w_mat, delimiter=',')
    #w_mat = np.loadtxt('data.csv',delimiter=',')
    test_word_similarity(w_mat,token_ids)
   
    d = DataLoader(lower_case=True, include_newsgroup=False)
    w2v_mat,w2v_token = vec_spaces.create_word2vec_matrix(d)
    test_word2vec_similarity(w2v_mat,w2v_token)
    
    
    ####### End of your code #############


if __name__ == "__main__":
    main()
