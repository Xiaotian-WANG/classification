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
             '位置', '分析', '状态', 'SUST', '轴承',
             '运输', '人员', '失效', '审定', '断裂',
             '损坏', '关闭', '跑道', '着陆', '安全局',
             '断开', '压力', '委员会', '信息', '进入',
             '传感器', '滑油', '控制', '检查单', '左侧',
             '飞行', '液压', '驾驶舱', '发动机', 'QRH']

keywords[1] = ['失去', '设计',  '机组', '人员', '压力',
             '机场',  '着陆', '信息', '使用', '组件',
             '前起落架', '导航', '襟翼', '中央', '问题',
             '严重', '发动机', '支架', 'BFU', '成功',
             '穿过', '飞行员','事故', '之间', '缺少',
             '飞行', '过程', '脱落', '扰流板', '服务',
             '公告', '两', '则', '保持', '推力', '新',
             '备用', '处置', '位置', '以及', '控制',
             '要求', '征候', '机翼', '计划','AAIB']

keywords[2] = ['大修', '维修', '过程', '主起落架', '整流罩',
             '锁', '安装',  '金属', '工作','检查','疲劳',
             '规定', '锁定', '次', '维护', '关闭','脱落',
             '航前', '镀', '右侧', '因为','失效', '工程师',
             '间隙', '起飞', '状态', '发现', '作用', '左侧',
             '机构', '表面',  '飞机', '安全', '要求', '闩锁',
             '维修人员', '执行','因素', '分离', '固定', '正确',
             '内侧', '风扇整流罩']

keywords[3] = ['机组', '起飞', '异常', '人员', '飞行', '顺桨',
             '未', '信息', '偏离',  '内部', '公司', '自动',
             '检查单', '沟通', '操作', '说明', '缺乏', '之间',
             '根据',  '发动机','滑跑', '决定', '坠机', '影响',
             '系统', '关闭', '飞行员', '考虑', '被', '执行',
             '因素', '失速', '问题', '右侧','航班', '显示',
             '分析', '程序', '飞机', '事故', 'CAO', '电子',
             '位置', '表面', '确定', '发现', '公里', '阶段',
             '驾驶舱', '随后', '起火', '来', '建议', '主要']

keywords[4] = ['制造', '燃油', '产生', '异常', '裂纹', '盘', '起火',
           '低循环', '涡轮盘', '左侧', '发动机', '疲劳裂纹',
           '起飞', '滑跑', 'HPT', '停止', '过程', '失效',
           '高压涡轮', '表面', '装配',  '地面', '使用', '关闭',
           '因素', '内部', '操作', '延迟', '发现',  '程序',
           '螺栓', '影响', '检查单', '泄漏', '加工', 'NTS',
           '切断', '供', '线', '主燃料', '箱', '高压', '生产', ]

keywords[5] = ['运营人', '公司', '制造商', '叶片', '失效', '区域',
             '系统', '故障', '航空', '状态', '组织', '非包容性',
             '襟翼', '机组', '右侧', '飞行', '发动机', '使用',
             '内', '人员', '手册', '民用航空', '监督', '评估',
             '影响', '程序','污染物', '下方', '飞机', '问题',
             '运行', '分析', '进入', '因素', 'CAO', '处置',
             '分离',  '传输', '腐蚀', '安装', '阶段', '技术']

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
