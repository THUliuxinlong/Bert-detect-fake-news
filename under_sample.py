import pandas as pd
import numpy as np

# 第三步：加载AEDA之后的数据
raw_data = pd.read_csv("./data/data_augs_1.csv")
print(raw_data.head())

# 取假新闻的索引
fakenews_indices = raw_data[raw_data['label'] == 0].index  # data['Class'] == 1 会返回一串 Ture False 字符串列表，再把这个当作索引
# 取真新闻的索引
realnews_indices = np.array(raw_data[raw_data['label'] == 1].index)
print(len(fakenews_indices))
print(len(realnews_indices))

# 随机生成假新闻的索引
random_fakenews_indices = np.random.choice(a=fakenews_indices, size=len(realnews_indices), replace=False)
print(len(random_fakenews_indices))

# 将class=1和class=0 的选出来的索引值进行合并 此时这两个样本的数量是一样的
under_sample_indices = np.concatenate([realnews_indices, random_fakenews_indices])

# 取对应索引的数据
under_sample_data = raw_data.iloc[under_sample_indices, :]
print(under_sample_data.head())
label_counts = under_sample_data.groupby("label").label.value_counts()
print(label_counts)

under_sample_data.to_csv('./data/under_sample_data.csv',index=False)


