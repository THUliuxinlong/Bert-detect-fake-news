import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from transformers import logging

# 关掉一个报错
logging.set_verbosity_error()

# 一些画图的参数
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# # load data
# raw_data = pd.read_csv("./data/train.csv")
# print(raw_data.head())
# print(raw_data.shape)

# # 第一步：去除特殊字符
# def clean_special_chars(text, punct):
#     for p in punct:
#         text = text.replace(p, '')
#     return text
#
# punct = ',，@#ð±¦­ââ¤ï¸¼¹µ¼©¹¾°ª»¶³ó' \
#         'æ´åå0616çæ¥å¿«ä®ã¢ëìê·_í¬£ ÂÑºÐ'
#
# no_punct_data = DataFrame({
#     'id':[],
#     'label':[],
#     'tweet':[]
# })
# no_punct_data['id'] = raw_data['id']
# no_punct_data['label'] = raw_data['label']
# no_punct_data['tweet'] = raw_data['tweet'].apply(lambda x: clean_special_chars(x, punct))
# no_punct_data.to_csv('./data/no_punct_data.csv',index=False)
# print(no_punct_data.head())

# no_punct_data = pd.read_csv("./data/no_punct_data.csv")

# data_augs_1 = pd.read_csv("./data/data_augs_1.csv")

under_sample_data = pd.read_csv("./data/under_sample_data.csv")

# Explore Data Analyze
# 看真假新闻的数量
label_counts = under_sample_data.groupby("label").label.value_counts()
print(label_counts)
sns.countplot(x=under_sample_data['label'])
plt.show()

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# 句长统计
token_lens = []
for txt in under_sample_data.tweet:
  tokens = tokenizer.encode(txt, max_length=512, truncation=True)
  token_lens.append(len(tokens))

sns.histplot(token_lens)
plt.xlim([0, 100])
plt.xlabel('Token count')
plt.show()


class_names = ['fake', 'real']
class NewsDataset(Dataset):

    def __init__(self, tweet, label, tokenizer, max_len):
        self.tweet = tweet
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        label = self.label[item]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'tweet': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# dataloader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = NewsDataset(
        tweet=df.tweet.to_numpy(),
        label=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

MAX_LEN = 40
BATCH_SIZE = 32
# 拆分数据
df_train, df_test = train_test_split(under_sample_data, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# 定义模型
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

class NewsClassifier(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = NewsClassifier(n_classes=2)
model = model.to(device)
# print(model)

# 训练
EPOCHS = 10
learning_rate = 2e-5

# AdamW优化器，它纠正了重量衰减，还将使用没有预热步骤的线性调度程序
optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

# 定义一次训练
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):

    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        label = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, label)

        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())

        loss.backward()
        # clip_grad_norm_裁剪模型的梯度来避免梯度爆炸。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# 评估模型
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, label)

            correct_predictions += torch.sum(preds == label)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# 训练循环并存储训练历史记录
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    # 转到cpu上，方便后面绘制
    train_acc_cpu = train_acc
    val_acc_cpu = val_acc
    history['train_acc'].append(train_acc_cpu.cpu())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc_cpu.cpu())
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.show()

# 测试集评估
test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
)

print('test_acc', test_acc.item())

# 和评估函数类似，但是存储了新闻的文本和预测概率
def get_predictions(model, data_loader):
    model = model.eval()

    tweet_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["tweet"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            tweet_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(label)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return tweet_texts, predictions, prediction_probs, real_values

y_tweet_texts, y_pred, y_pred_probs, y_real_label = get_predictions(
    model,
    test_data_loader
)

print(classification_report(y_real_label, y_pred, target_names=class_names))

# 绘制混淆矩阵
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted')
    plt.show()

cm = confusion_matrix(y_real_label, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)


# # 测试一个实例
# idx = 2
# review_text = y_tweet_texts[idx]
# true_sentiment = y_real_label[idx]
# pred_df = pd.DataFrame({
#     'class_names': class_names,
#     'values': y_pred_probs[idx]
# })
#
# print("\n".join(wrap(review_text)))
# print()
# print(f'True sentiment: {class_names[true_sentiment]}')
#
# # 真假的置信度
# sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
# plt.ylabel('sentiment')
# plt.xlabel('probability')
# plt.xlim([0, 1])
# plt.show()
#
# # 对一句话进行测试
# review_text = "I love completing my todos! Best app ever!!!"
#
# encoded_review = tokenizer.encode_plus(
#     review_text,
#     max_length=MAX_LEN,
#     add_special_tokens=True,
#     return_token_type_ids=False,
#     pad_to_max_length=True,
#     return_attention_mask=True,
#     return_tensors='pt',
# )
#
# input_ids = encoded_review['input_ids'].to(device)
# attention_mask = encoded_review['attention_mask'].to(device)
#
# output = model(input_ids, attention_mask)
# _, prediction = torch.max(output, dim=1)
#
# print(f'Review text: {review_text}')
# print(f'Sentiment  : {class_names[prediction]}')