import re

mykeywords = ["发动机","控制","失效","部件","故障","系统","航空器","疲劳","断裂","泄漏","失灵"]

x_text = [""]
with open('data.txt', 'r',encoding='utf-8') as datafile:
    lines = datafile.readlines()
    for line in lines:
        if(line != '\n'):
            line = re.sub('[\n]','',line)
            line = re.sub('["]','',line)
            x_text[-1] += line
        else:
            x_text.append('')

with open('y_1.txt', 'r') as datafile:
    y1_label = datafile.readlines()
    for i in range(len(y1_label)):
        y1_label[i] = re.sub('[\n]','',y1_label[i])
        y1_label[i] = int(y1_label[i])

def get_vector(text,keywords):
    vector = list()
    for keyword in keywords:
        vector.append(text.count(keyword))
    return vector

print(get_vector(x_text[0],mykeywords))

x_vector = []
for x in x_text:
    x_vector.append(get_vector(x,mykeywords))

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array(x_vector)
y = np.array(y1_label)

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)

