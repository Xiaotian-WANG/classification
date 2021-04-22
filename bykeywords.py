import numpy as np

keywords = {}

keywords[0] = ["系统","部件","失效","故障",
"断裂","损坏","关闭","断开",
"控制","传感器","失灵","泄露"]

keywords[1] = ["设计","制造","缺陷","组件",
"BFU","AAIB"]

keywords[2] = ["大修","维修","安装","检查",
"机务","工程师","维修人员",
"执行","维护","人员","检查单","替换"]

keywords[3] = ["机组","人员","操作","飞行员",
"决定","失速","沟通","根据",
"滑跑","CAO","驾驶","驾驶舱"]

keywords[4] = ["制造","质量","裂纹","疲劳裂纹",
"疲劳","泄露","生产","加工",
"装配","部件","零件","零部件"]

keywords[5] = ["运营","管理","运营人","公司",
"制造商","组织","监督","评估",
"程序","非包容性"]

keywords[6] = ["程序","规定","检查单","规章",
"手册","检查","监控","安全","公司","指示"]

def clf(X,i):
    cls = []
    for x in X:
        count = 0
        for word in keywords[i]:
            count+=x.count(word)
        if count >= 3:
            cls.append(1)
        else:
            cls.append(0)
    return cls

X = np.load("X_lac.npy", allow_pickle=True)
y = np.load("y_lac.npy")

myclass = 6

y = y[myclass]

cls = np.array(clf(X,myclass))
print(cls)


print((cls == y)+0)

print("accuracy: ", sum(cls==y)/cls.size)
print("precision: ",sum(((cls==y)+0)*((cls==1)+0))/sum(y==1))
print("recall: ", sum(((cls==y)+0)*((cls==1)+0))/(sum(((cls==y)+0)*((cls==1)+0))+(sum(((cls!=y)+0)*((y==0)+0)))))
