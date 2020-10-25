import numpy as np
from gensim.models import Word2Vec



def create_word2vec_matrix(data_loader, min_count=20, size=100, window_size=3):
    """

    Arguments
    ---------
    `data_loader`: `list`-like
        A `list` (or `list` like object), each item of which is itself a list of tokens.
        For example:
                
                [
                    ['this', 'is', 'sentence', 'one'],
                    ['this', 'is', 'sentence', 'two.']
                ]

        Note that preprocesisng.DataLoader is exactly this kind of object.
    `min_count`: `int`
        The minimum count that a token has to have to be included in the vocabulary.
    `size`: `int`
        The dimensionality of the word vectors.
    `window_size`: `int`
        The window size. Read the assignment pdf if you don't know what that is.

    Returns
    -------
        `tuple(np.ndarray, dict)`:
            The first element will be an (V, `size`) matrix, where V is the
            resulting vocabulary size.
            
            The second element is a mapping from `int` to `str` of which word
            in the vocabulary corresponds to which index in the matrix. 
    """

    word2vec_mat = None

    word2vec_id_to_tokens = {}
    ####### Your code here ###############
    model = Word2Vec(data_loader, min_count=min_count,size= size,workers=3, window =window_size, sg = 1)
    model.save('model.model')
    #model = Word2Vec.load("model.model")
    vocab = list(model.wv.vocab)
    word2vec_mat = model.wv.vectors
    i = 0
    for item in vocab:
        word2vec_id_to_tokens[i] = item
        i =i+1
    ####### End of your code #############

    return word2vec_mat, word2vec_id_to_tokens


def create_term_newsgroup_matrix(
    newsgroup_and_token_ids_per_post,
    id_to_tokens,
    id_to_newsgroups,
    tf_idf_weighing=False,
):
    """

    Arguments
    ---------
        newsgroup_and_token_ids_per_post: `list`
            Each item will be a `tuple` of length 2.
            Each `tuple` contains a `list` of token ids(`int`s), and a newsgroup id(`int`)
            Something like this:
                
                newsgroup_and_token_ids_per_post=[
                    ([0, 54, 3, 6, 7, 7], 0),
                    ([0, 4,  7], 0),
                    ([0, 463,  435, 656,  ], 1),
                ]

            The "newsgroup_and_token_ids_per_post" that main.read_processed_data() returns
            is exactly in this format.
        
        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello"  ...}

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.
            
                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        tf_idf_weighing: `bool`
            Whether to use TF IDF weighing in returned matrix.

    Returns
    -------
        `np.ndarray`:
            Shape will be (len(id_to_tokens), len(id_to_newsgroups)).
            That is it will be a VxD matrix where V is vocabulary size and D is number of newsgroups.

            Note, you may choose to remove the row corresponding to the "UNK" (which stands for unknown)
            token.
    """

    V = len(id_to_tokens)
    D = len(id_to_newsgroups)
    #print(V,D)
    mat = np.zeros((V,D))
    #print(mat.shape)
    ####### Your code here ###############
    for item in newsgroup_and_token_ids_per_post:
        ngroup = id_to_newsgroups[item[1]]
        token = item[0]
        for tok_id in token:
            mat[tok_id][item[1]] = mat[tok_id][item[1]] + 1
    ####### End of your code #############

    if tf_idf_weighing:
        # mat_trim = np.copy(mat[:10,:])
        tf_mat = np.copy(mat)
        tf_mat[tf_mat > 0] = 1 + np.log(tf_mat[tf_mat > 0])
        df = (mat != 0).sum(1)
        idf_mat = np.log(len(id_to_newsgroups)/df)
        idf_mat = np.reshape(idf_mat, (idf_mat.shape[0], 1))
        tf_idf_mat = tf_mat * idf_mat
        # print(tf_idf_mat[0:5])
        # print(id_to_tokens[2])
        # print(id_to_tokens[4])
        return tf_idf_mat

    return mat


def create_term_context_matrix(
    newsgroup_and_token_ids_per_post,
    id_to_tokens,
    id_to_newsgroups,
    ppmi_weighing=False,
    window_size=5,
):
    """

    Arguments
    ---------
        newsgroup_and_token_ids_per_post: `list`
            Each item will be a `tuple` of length 2.
            Each `tuple` is a post, contains a `list` of token ids(`int`s), and a newsgroup id(`int`)
            Something like this:
                
                newsgroup_and_token_ids_per_post=[
                    ([0, 54, 3, 6, 7, 7], 0),
                    ([0, 4,  7], 0),
                    ([0, 463,  435, 656,  ], 1),
                ]

            The "newsgroup_and_token_ids_per_post" that main.read_processed_data() returns
            is exactly in this format.
        
        id_to_tokens: `dict`
            `int` token ids as keys and `str` tokens as values.
            
                { 0: "hi", 1: "hello  ...}

        id_to_newsgroups: `dict`
            `int` newsgroups ids as keys and `str` newsgroups as values.
            
                { 0: "hi", 1: "hello  ...}
                { 0: "comp.graphics", 1: "comp.sys.ibm.pc.hardware" .. }

        ppmi_weighing: `bool`
            Whether to use PPMI weighing in returned matrix.

    Returns
    -------
        `np.ndarray`:
            Shape will be (len(id_to_tokens), len(id_to_tokens)).
            That is it will be a VxV matrix where V is vocabulary size.

            Note, you may choose to remove the row/column corresponding to the "UNK" (which stands for unknown)
            token.
    """

    V = len(id_to_tokens)
    mat = np.zeros((V,V))
    ####### Your code here ###############
    count = 0
    for item in newsgroup_and_token_ids_per_post:
       
        item = item[0]
        for i in range(len(item)):
            low = i - window_size
            if(low<0):
                low = 0
            upper = i + window_size
            for j in range(len(item)):
                if(j>=low and j<=upper):
                    if item[i] == item[j]:
                        #print('is all good',item[i],item[j])
                        mat[item[i]][item[j]] = 0
                    else:
                        mat[item[i]][item[j]] += 1

    if ppmi_weighing:
        ppmi_mat = np.copy(mat)
        total = np.sum(ppmi_mat)
        col_sum = ppmi_mat.sum(axis=0)/total
        row_sum = ppmi_mat.sum(axis=1)/total

        row_sum = np.reshape(row_sum, (row_sum.shape[0], 1))
        col_sum = np.reshape(col_sum, (1, col_sum.shape[0]))
        
        ppmi_mat = ppmi_mat/total
        ppmi_mat = ppmi_mat/row_sum
        ppmi_mat[np.isnan(ppmi_mat)] = 0
        ppmi_mat = ppmi_mat/col_sum
        ppmi_mat[np.isnan(ppmi_mat)] = 0
        ppmi_mat[ppmi_mat == 0] = 0.00000001
        ppmi_mat = np.log(ppmi_mat)
        ppmi_mat[ppmi_mat < 0] = 0
        #print(ppmi_mat)
        return ppmi_mat

    
    ####### End of your code #############
    return mat


def compute_cosine_similarity(a, B):
    """Cosine similarity.

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The cosine similarity between a and every
            row in B.
    """
    vals = []
    ####### Your code here ###############
    for b in B:
        cos = np.dot(a,b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b)))
        vals.append(cos)
    return vals
    ####### End of your code #############


def compute_jaccard_similarity(a, B):
    """

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The Jaccard similarity between a and every
            row in B.
    """
    ####### Your code here ###############
    vals = []
    for b in B:
        s1 = set(a)
        s2 = set(b)
        sim = len(s1.intersection(s2)) / len(s1.union(s2))
        vals.append(sim)
    ####### End of your code #############
    return vals

def compute_dice_similarity(a, B):
    """
stolen : 0.17904197
weapons : 0.17657003
and : 0.27388635
answer : 0.12892626
no : 0.1688911
it : 0.32679212
s : 0.19813703
total : 0.28250358
good : 0.22370566
point : 0.0052258903

    Arguments
    ---------
        a: `np.ndarray`, (M,)
        B: `np.ndarray`, (N, M)

    Returns
    -------
        `np.ndarray` (N,)
            The Dice similarity between a and every
            row in B.
    """
    val = []
    ####### Your code here ###############
    for b in B:
        intersection = np.logical_and(a, b)
        sim = 2 * intersection.sum() / (a.sum() + b.sum())
        val.append(sim)
    ####### End of your code #############
    return val
