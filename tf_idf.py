from math import log10
import numpy as np

def tf(text,document):
    return  log10(document.count(text)+1)

def df(texts, documents):
    if type(texts) == str:
        texts = list(texts)
    dfs = []
    for text in texts:
        df = 0
        for document in documents:
            if (text in document):
                df += 1
        dfs.append(log10(df))
    return dict(zip(texts,dfs))

def idf(text, documents):
    df = 0
    for document in documents:
        if (text in document):
            df += 1
    idf = len(documents)/df
    return log10(idf)

def tf_idf(texts, document, documents):
    if type(texts) == str:
        texts = list(texts)

    tfs = np.array([])
    for text in texts:
        tfs = np.append(tfs,tf(text,document))
    idfs = np.array([])
    for text in texts:
        idfs = np.append(idfs, idf(text, documents))

    return dict(zip(texts, tfs*idfs))
