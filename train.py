import numpy as np

X = np.load("X_lac.npy", allow_pickle=True)
y = np.load("y_lac.npy")


def get_vector(text,keywords):
    vector = list()
    for keyword in keywords:
        vector.append(text.count(keyword))
    return vector


keywords = [[],[],[],[],[],[],[]]

keywords[0] = ['支柱', '故障', 'ATS', '机组', '起落架',
             '状态', 'SUST', '轴承',
             '运输', '失效', '断裂',
             '损坏', '关闭', '跑道', '着陆',
             '断开', '压力', '信息',
             '传感器', '滑油', '控制', '发动机']

keywords[1] = ['设计',
             '组件',
             '前起落架', '导航', '襟翼',
             '发动机', '支架', 'BFU',
             '脱落', '扰流板',
             '保持', '推力',
             '处置', '位置',
             '征候', '机翼', '计划', 'AAIB']

keywords[2] = ['大修', '维修', '过程', '主起落架', '整流罩',
             '安装', '工作', '检查', '疲劳',
             '规定', '锁定', '维护',
             '失效', '工程师',
             '间隙', '起飞', '状态', '左侧',
             '机构', '表面', '要求', '闩锁',
             '维修人员', '执行', '固定',
             '内侧', '风扇整流罩']

keywords[3] = ['机组', '起飞', '异常', '人员', '顺桨',
             '偏离',  '内部', '公司', '自动',
             '检查单', '沟通', '操作',
             '根据',  '发动机','滑跑', '决定', '坠机',
             '飞行员', '考虑',
             '失速', '问题', '右侧','航班',
             '分析', '事故', 'CAO', '电子',
             '位置', '表面', '发现',  '阶段',
             '驾驶舱', '起火']

keywords[4] = ['制造', '燃油', '产生', '异常', '裂纹', '起火',
           '低循环', '涡轮盘',  '发动机', '疲劳裂纹',
           'HPT', '停止', '过程', '失效',
           '高压涡轮', '表面', '装配',
           '螺栓', '检查单', '泄漏', '加工', 'NTS',
           '切断', '主燃料', '箱', '高压', '生产']

keywords[5] = ['运营人', '公司', '制造商', '叶片', '失效', '区域',
             '系统', '故障', '航空', '状态', '组织', '非包容性',
             '襟翼', '右侧', '发动机',
             '人员', '手册',  '监督', '评估',
             '影响', '程序','污染物', '下方', '飞机', '问题',
             '运行', '分析', '进入', '因素', 'CAO', '处置',
             '传输', '安装', '技术']

keywords[6] = ['释放', '缺少', '程序', '缺乏', '规定', '检查单',
             '延迟', '人员', '机组', '中断', '严重', '着陆',
             '要求', '检查', '使用', '发动机', '动力', '飞行',
             '过程', '起火', '内部', '燃油', '齿轮箱', '故障',
             '因素', '产生', '停止', '监控', '影响', '继续',
             '安全', '左侧', '意识', '指示', '机场', '轴承',
             '公司', '检测', 'QRH', '决定', '安装']

allkeywords = []
for i in range(7):
    allkeywords += keywords[i]

allkeywords = set(allkeywords)

allkeywords = [
"系统","部件","失效","故障",
"断裂","损坏","关闭","断开",
"控制","传感器","失灵","泄露",

"设计","制造","缺陷","组件",
"BFU","AAIB",

"大修","维修","安装","检查",
"机务","工程师","维修人员",
"执行","维护","人员","检查单","替换",

"机组","人员","操作","飞行员",
"决定","失速","沟通","根据",
"滑跑","CAO","驾驶","驾驶舱",

"制造","质量","裂纹","疲劳裂纹",
"疲劳","泄露","生产","加工",
"装配","部件","零件","零部件",

"运营","管理","运营人","公司",
"制造商","组织","监督","评估",
"程序","非包容性",

"程序","规定","检查单","规章",
"手册","检查","监控","安全",
"公司","指示"
]


x_vector = []
for x in X:
    x_vector.append(get_vector(x, allkeywords))

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

'''
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
x_vector = pca.fit(np.array(x_vector).transpose()).components_.transpose()
for item in x_vector:
    item = item.tolist()
x_vector = x_vector.tolist()
'''


X_train, X_test, y_train, y_test = train_test_split(np.array(x_vector), y[0], test_size=0.2)


rate = int((y_train.size-y_train.sum())/y_train.sum())
index = np.where(y_train == 1)
X_train1 = X_train[index]
y_train1 = y_train[index]
X_train1 = np.tile(X_train1, (rate,1))
y_train1 = np.tile(y_train1, rate)
X_train = np.vstack((X_train, X_train1))
y_train = np.hstack((y_train, y_train1))

X_train, y_train = shuffle(X_train, y_train)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

loss = np.power(clf.predict(X_train)-y_train, 2).sum()/y_train.size
res1 = clf.predict(X_test)
print(res1)




from sklearn.naive_bayes import GaussianNB

clf1 = GaussianNB()
clf1.fit(X_train,y_train)
loss1 = np.power(clf1.predict(X_train)-y_train, 2).sum()/y_train.size
res2 = clf1.predict(X_test)
print(res2)

from sklearn.linear_model import LogisticRegression

clf2 = LogisticRegression(penalty='l1', solver='liblinear')
clf2.fit(X_train,y_train)
loss2 = np.power(clf2.predict(X_train)-y_train, 2).sum()/y_train.size
res3 = clf2.predict(X_test)
print(res3)
print("")
res = ((res1+res2+res3)>=2)+0
print(res)
print(y_test)

print("accuracy: ", np.sum(res==y_test)/y_test.size)
print("baseline: ",1-y_test.sum()/y_test.size)
