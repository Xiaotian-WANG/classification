import pandas
# import thulac
import numpy as np



file_path = "data.xlsx"

data = pandas.read_excel(file_path, sheet_name="原始数据",engine="openpyxl")

X = data["原因总结"]

y = data["事故原因（大类）"]

myclasses_dict = {"航空器系统/部件失效":1,
             "航空器设计制造缺陷":2,
             "机务人员致灾":3,
             "机组人员致灾":4,
             "零件生产质量问题":5,
             "运营管理问题":6,
             "程序规章手册缺陷":7,
             "安检空管维修资质等其它":8}

myclasses = ["航空器系统/部件失效",
             "航空器设计制造缺陷",
             "机务人员致灾",
             "机组人员致灾",
             "零件生产质量问题",
             "运营管理问题",
             "程序规章手册缺陷",
             "安检空管维修资质等其它"]
from ltp import LTP
ltp = LTP()

X_dataset, _ = ltp.seg(X.tolist())
X_dataset = np.array(X_dataset, dtype=object)
y_dataset = [[],[],[],[],[],[],[],[]]

for i in range(len(myclasses)):
    for item in y.tolist():
        if myclasses[i] in item:
           y_dataset[i].append(1)
        else:
            y_dataset[i].append(0)

y_datatset = np.mat(y_dataset)

np.save("X_ltp.npy", X_dataset)
np.save("y_ltp.npy",y_dataset)
