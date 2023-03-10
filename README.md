# 智能信息第二次实验

[TOC]

## 1、Explore Data Analyze

### 1. 统计真假新闻的数量

``` python
label
0        29720
1         2242
```

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121534.png" alt="image-20230307121533708" style="zoom:33%;" />

可以看出数据的类别极不均衡，需要在数据预处理阶段进行改进。

### 2. 统计句长

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121518.png" alt="image-20230307121517649" style="zoom:33%;" />

可以看出tweet的长度大多集中在40以下，后面将数据裁剪成40个token。

## 2、数据处理

因为bert模型自带词典很强大，不需要对文本进行很多处理。我这里主要针对文本里的特殊字符进行过滤，对类别不均衡问题进行解决。

### 1. 去除特殊字符

#### 处理前

> @user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   #run
>
> @user @user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx.    #disapointed #getthanked
>
> bihday your majesty
>
> #model   i love u take with u all the time in urð±!!! ðððð¦ð¦ð¦  

#### 处理后

>  user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   run
>
> user user thanks for lyft credit i can't use cause they don't offer wheelchair vans in pdx.    disapointed getthanked  
>
> bihday your majesty
>
> model   i love u take with u all the time in ur!!!  

### 2、AEDA

使用AEDA的前后对比。

> id,label,tweet
>
> 1,0, ! user when a father is ? dysfunctional and ! is so selfish he ; drags , his kids ; into , his dysfunction.   run
>
> 1,0, user when a father is dysfunctional and is so selfish he drags his kids into his dysfunction.   run
>
> 2,0,user user thanks ! for ; lyft credit i can't use cause they don't offer wheelchair vans in pdx.   :  disapointed getthanked
>
> 2,0,user user thanks for lyft credit i can't use cause they don't offer wheelchair vans in pdx.    disapointed getthanked
>
> 3,0,  . bihday your majesty
>
> 3,0,  bihday your majesty

### 3.处理类别不均衡

对数据进行下采样，将真假新闻的数量大致统一。

``` python
raw_label
0        29720
1         2242
AEDA_label
0        59440
1         4484
AEDA_label
0        4484
1         4484
```

## 4、模型搭建与训练

### 1、模型结构

```python
NewsClassifier(
  (bert): BertModel
  (drop): Dropout(p=0.3, inplace=False)
  (out): Linear(in_features=768, out_features=2, bias=True)
)
```

### 2、模型训练

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121556.png" alt="image-20230307121555172" style="zoom:33%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121612.png" alt="image-20230307121611169" style="zoom:33%;" />

``` python
test_acc 0.9668542839274546
              precision    recall  f1-score   support

        fake       0.98      0.99      0.98      1497
        real       0.80      0.65      0.71       102

    accuracy                           0.97      1599
   macro avg       0.89      0.82      0.85      1599
weighted avg       0.96      0.97      0.97      1599
```

#### 加入AEDA后

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121633.png" alt="image-20230307121632115" style="zoom:33%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121829.png" alt="image-20230307121828639" style="zoom:33%;" />

``` python
test_acc 0.9965592743196747
              precision    recall  f1-score   support

        fake       1.00      1.00      1.00      2977
        real       1.00      0.95      0.97       220

    accuracy                           1.00      3197
   macro avg       1.00      0.98      0.99      3197
weighted avg       1.00      1.00      1.00      3197
```

相较于不使用AEDA精确率、召回率、F1-score均有提升。在测试集的准确率提升了3%左右。

#### 数据下采样+AEDA

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121706.png" alt="image-20230307121705023" style="zoom:33%;" />

<img src="https://lxlpicbed.oss-cn-beijing.aliyuncs.com/img/2023-03-07-121721.png" alt="image-20230307121720784" style="zoom:33%;" />

``` python
Epoch 10/10
----------
Train loss 0.00010004409306980357 accuracy 1.0
Val   loss 0.48373106205053773 accuracy 0.9486607142857142

test_acc 0.9665924276169264
              precision    recall  f1-score   support

        fake       0.99      0.95      0.97       236
        real       0.95      0.99      0.97       213

    accuracy                           0.97       449
   macro avg       0.97      0.97      0.97       449
weighted avg       0.97      0.97      0.97       449
```

BERT 在针对特定任务的小型语料库进行微调时表现良好，即使是对于该实验13：1的类别不均衡也能达到很好的效果。

由于下采样，使数据量减少，导致丢失了许多有用信息，也导致模型在训练集上过拟合，效果反而比只用AEDA要差。

