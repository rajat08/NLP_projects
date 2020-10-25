from nltk.corpus import conll2002



train_sents = list(conll2002.iob_sents("esp.train"))
dev_sents = list(conll2002.iob_sents("esp.testa"))
test_sents = list(conll2002.iob_sents("esp.testb"))

with open("results.txt", "w") as out:

    for sent in train_sents:
        s = ''
        for i in range(len(sent)):
            w = sent[i][0]
            s =s + w + ' '
        #print(s)
        out.write(s)
        out.write('\n')
    for sent in dev_sents:
        s = ''
        for i in range(len(sent)):
            w = sent[i][0]
            s =s + w + ' '
        #print(s)
        out.write(s)
        out.write('\n')
    for sent in test_sents:
        s = ''
        for i in range(len(sent)):
            w = sent[i][0]
            s =s + w + ' '
        #print(s)
        out.write(s)
        out.write('\n')
    