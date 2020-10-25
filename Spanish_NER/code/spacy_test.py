import spacy

nlp = spacy.load('es_core_news_md')
t = nlp('la sentencia')
print(t[0].vector)
