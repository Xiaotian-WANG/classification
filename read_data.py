import pandas

file_path = "data.xlsx"

data = pandas.read_excel(file_path, sheet_name="原始数据",engine="openpyxl")

X = data["原因总结"]

y = data["事故原因（大类）"]

myclasses = ["航空器系统/部件失效",
             "航空器设计制造缺陷",
             "机务人员致灾",
             "机组人员致灾",
             "零件生产质量问题",
             "运营管理问题",
             "程序规章手册缺陷",
             "安检空管维修资质等其它"]

