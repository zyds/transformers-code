# %% [markdown]
# # 文本分类实例

# %% [markdown]
# ## Step1 导入相关包

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification

import torch.distributed as dist

dist.init_process_group(backend="nccl")

# %% [markdown]
# ## Step2 加载数据

# %%
import pandas as pd

data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
data

# %%
data = data.dropna()
data

# %% [markdown]
# ## Step3 创建Dataset

# %%
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)

# %%
dataset = MyDataset()
# for i in range(5):
#     print(dataset[i])

# %% [markdown]
# ## Step4 划分数据集

# %%
import torch
from torch.utils.data import random_split


trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
len(trainset), len(validset)

# %%
for i in range(5):
    print(trainset[i])

# %% [markdown]
# ## Step5 创建Dataloader

# %%
import torch

tokenizer = BertTokenizer.from_pretrained("/gemini/code/model")

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

# %%
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))

# %%
next(enumerate(validloader))[1]

# %% [markdown]
# ## Step6 创建模型及优化器

# %%
from torch.optim import Adam
import os
from torch.nn.parallel import DistributedDataParallel as DDP

model = BertForSequenceClassification.from_pretrained("/gemini/code/model")

if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))

model = DDP(model)

# %%
optimizer = Adam(model.parameters(), lr=2e-5)

# %% [markdown]
# ## Step7 训练与验证

def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:
        print(info)

# %%
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    dist.all_reduce(acc_num)
    return acc_num / len(validset)

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        print_rank_0(f"ep: {ep}, acc: {acc}")

# %% [markdown]
# ## Step8 模型训练

# %%
train()