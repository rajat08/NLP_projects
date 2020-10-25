"""Experiments with scraped wikipedia data and NLP tools
"""
import en_core_web_sm
import json
import spacy
from pathlib import Path
from collections import  defaultdict

# The path that WikiExtractor.py will extract to
JSONL_FILES_DIR = Path("text/AA")


def yield_all_articles(path: Path = JSONL_FILES_DIR):
    """Go through all the extracted wikipedia files and "yield" one at a time.
    
    Read about generators here if necessary: https://wiki.python.org/moin/Generators
    """
    for one_file in path.iterdir():
        print("Going through", one_file.name)
        with open(one_file) as f:
            for line in f:
                text = json.loads(line)["text"]
                yield text
                break


def count_things():

    article_gen = yield_all_articles(JSONL_FILES_DIR)
    print("About to load spacy model")
    nlp = spacy.load('en_core_web_sm')
    print("Finished loading spacy's English models")


    ###### Write below #########

    tc = 0
    lem = []
    lemc = 0
    ner = defaultdict(list)
    for article in article_gen:
        doc = nlp(article)
        for token in doc:
            if not token.is_punct:
                tc = tc+1
            lem.append(token.lemma_)
        lset = set(lem)
        lemc = lemc + len(lset)
        for ent in doc.ents:
            ner[ent.label_].append(ent.text)
     
    
    non_punct_non_space_token_count = tc
    
    lemma_count = lemc
    for ntype,nlist in ner.items():
        ner[ntype] = len(nlist)
    ner_count_per_type = ner

    ###### End of your work #########

    print(
        "Non punctuation and non space token count: {}\nLemma count: {}".format(
            non_punct_non_space_token_count, lemma_count
        )
    )
    print("Named entity counts per type of named entity:")
    for ner_type, count in ner_count_per_type.items():
        print("{}: {}".format(ner_type, count))


if __name__ == "__main__":
    count_things()
