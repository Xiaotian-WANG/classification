import numpy as np
from zhon import hanzi
import string
from tf_idf import tf_idf, tf, idf, df

myclasses_dict = {"航空器系统/部件失效":0,
             "航空器设计制造缺陷":1,
             "机务人员致灾":2,
             "机组人员致灾":3,
             "零件生产质量问题":4,
             "运营管理问题":5,
             "程序规章手册缺陷":6,
             "安检空管维修资质等其它":7}

def get_frequency(X):
    word_frequency = {}

    for word in X:
        if word in hanzi.punctuation:
            pass
        elif word in string.punctuation:
            pass
        elif word == " ":
            pass
        else:
            word_frequency[word] = word_frequency.get(word, 0) + 1

    word_frequency = list(word_frequency.items())
    word_frequency.sort(key=lambda x: x[1], reverse=True)
    return word_frequency

X = np.load("X_lac.npy", allow_pickle=True)
y = np.load("y_lac.npy")


num = myclasses_dict["程序规章手册缺陷"]

wholetext1 = X[np.where(y[num] == 1)]
wholetext = []
for doc in wholetext1:
    wholetext = wholetext+doc

therest = np.delete(X,np.where(y[num]==1))



freq = get_frequency(wholetext)

wordlist = list(dict(freq).keys())

documents = [wholetext]
for item in therest:
    documents.append(item)

a = tf_idf(wordlist, wholetext, documents)
b = df(wordlist, wholetext1)

c = dict(zip(a.keys(), np.array(list(a.values()))*np.array(list(b.values())).tolist()))

c = list(c.items())
c.sort(key=lambda x: x[1], reverse=True)
c = dict(c)
