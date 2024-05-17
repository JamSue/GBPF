from datetime import datetime
import os
import csv
import pandas as pd
import numpy as np
import logging
import random
from collections import Counter
import torch
from tqdm import tqdm
import torch.nn.functional as F
from config import args,config_data

def Vote_Neark_label(out,classifer, center, labels, k):
    """
    out: 需要进行投票的样本向量 [batch_size, hidden_dim]
    classifer:模型的分类器部分
    center:当前保存在文件中的粒球中心
    labels:当前保存在文件中的粒球中心对应的标签
    k: 从k个粒球中心中选1个
    
    需要分别投出每条样本对应的中心再进行预测
    """ 
    #### 欧式距离
    # dis = torch.cdist(out, center, p=2) # 计算距离 
   
    #### 余弦距离，每个句子样本与中心向量分别求距离
    device = args.device
    out,center,labels = out.to(device),center.to(device),labels.to(device)
    
    if device=='cpu':
        dis =  (torch.zeros((out.shape[0], center.shape[0]))).to(device)
        for i in range(out.shape[0]):
            for j in range(center.shape[0]):
                similarities = F.cosine_similarity(out[i, :].unsqueeze(0), center[j, :].unsqueeze(0))
                dis[i, j] = 1 - similarities
    else:
        similarities = F.cosine_similarity(out.unsqueeze(1), center.unsqueeze(0), dim=2)
        dis = 1 - similarities
    
    topk_values, topk_indices = torch.topk(dis, k, largest=False) # 得到最小的k个中心的索引和k个距离,  topk_indices:[bs,k]
    
    # topk_labels = ([int(item) for item in labels[topk_indices]] )# 距离最近所对应的标签
    
    pred_label = torch.zeros(topk_indices.shape[0],dtype=torch.int) # 初始化batch_size个标签
    logits = torch.zeros(topk_indices.shape[0],config_data[args.dataset].labels_num)
    
    for idx in range(topk_indices.shape[0]): # 遍历每条样本的topk，找到对应的k个球心
        topk_embeddings = ([item for item in center[topk_indices[idx]]] )# 距离最近的k个embedding
        
        for i in range(k): # k个粒球中心过分类器后的logit
            if args.model_train==False:
                with torch.no_grad():
                    logit = classifer(topk_embeddings[i]) if (i==0) else (logit+ classifer(topk_embeddings[i]))
            else:
                logit = classifer(topk_embeddings[i]) if (i==0) else (logit+ classifer(topk_embeddings[i]))
        pred_label[idx] = torch.argmax(logit)
        logits[idx] = logit
        # print("###", classifer.state_dict())
    return logits, pred_label

### 找到最近的粒球中心，并返回中心向量和下标=>用于训练
'''
out: (bs, hidden_dim)
center(ball_num, hidden_dim)
'''
def find_nearest_center( out, center):
    
    device = args.device
    out, center = out.to(device),center.to(device)
    
    #### 欧式距离
    # dis = torch.cdist(out, center, p=2) # 返回形状为(5,3)  欧式距离
    
    #### 余弦距离，每个句子样本与中心向量分别求距离
    dis = (torch.zeros((out.shape[0], center.shape[0]))).to(device)
    for i in range(out.shape[0]):
        for j in range(center.shape[0]):
            similarities = F.cosine_similarity(out[i:i+1, :], center[j:j+1, :])
            dis[i, j] = 1 - similarities
    
    min_dis, indice = torch.min(dis,dim=1) # 寻找最近的一个粒球下标
    # topk_values, topk_indices = torch.topk(min_dis, k, largest=False)
    nearest_centers = center[indice]

    return nearest_centers,indice

def Vote_Neark(out,classifer, center, labels, k):
    """
    用于攻击，投出2k个最近的粒球中心，随机选出k个粒球中心反馈给攻击
    """ 
    #### 欧式距离
    # dis = torch.cdist(out, center, p=2) # 计算距离 
   
    #### 余弦距离，每个句子样本与中心向量分别求距离
    device = args.device
    out,center,labels = out.to(device),center.to(device),labels.to(device)
    
    if device=='cpu':
        dis =  (torch.zeros((out.shape[0], center.shape[0]))).to(device)
        for i in range(out.shape[0]):
            for j in range(center.shape[0]):
                similarities = F.cosine_similarity(out[i, :].unsqueeze(0), center[j, :].unsqueeze(0))
                dis[i, j] = 1 - similarities
    else:
        similarities = F.cosine_similarity(out.unsqueeze(1), center.unsqueeze(0), dim=2)
        dis = 1 - similarities
    
    topk_values, topk_indices = torch.topk(dis, 2*k, largest=False) # 得到最小的2k个中心的索引和2k个距离,  topk_indices:[bs,2k]
    
    # topk_labels = ([int(item) for item in labels[topk_indices]] )# 距离最近所对应的标签
    
    pred_label = torch.zeros(topk_indices.shape[0],dtype=torch.int) # 初始化batch_size个标签
    logits = torch.zeros(topk_indices.shape[0],config_data[args.dataset].labels_num)
    
    for idx in range(topk_indices.shape[0]): # 遍历每条样本的topk，找到对应的k个球心
        topk_embeddings = ([item for item in center[topk_indices[idx]]] )# 距离最近的k个embedding
        np.random.seed(110)
        index = np.random.choice(k,k//2,replace=False)
        for i in index: # k个粒球中心过分类器后的logit
            logit = classifer(topk_embeddings[i]) if (i==index[0]) else (logit+ classifer(topk_embeddings[i]))
        pred_label[idx] = torch.argmax(logit)
        logits[idx] = logit
    return logits, pred_label

### (4.1)保存最好的模型   
def save_checkpoint(best_accuracy,model_name,dataset_name, model, optimizer, epoch):
    logging.info('Best Model Saving...')
    model_state_dict = model.state_dict()
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save({
        'model_state_dict': model_state_dict,  # 网络参数
        'global_epoch': epoch,  # 最优对应epoch
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
        'best_acc': best_accuracy,  # 最好准确率
    }, os.path.join('checkpoints', 'model_{}_{}_best_{}.pth'.format(model_name, dataset_name,best_accuracy)))
    
### (7)数据处理的函数： 分词、句子转tokens
def str2tokens(sentence:str, tokenizer)->str:
    return tokenizer(sentence)

def tokens2seq(tokens:str, vocab, maxlen:int)->torch.Tensor:
    pad_word = 0
    x = [pad_word for _ in range(maxlen)]
    temp = tokens[:maxlen]
    for idx, word in enumerate(temp):
        x[idx] = vocab.get_index(word)
    return torch.tensor(x)

def str2seq(sentence:str, vocab, tokenizer, maxlen:int):
    tokens = str2tokens(sentence, tokenizer) 
    return tokens2seq(tokens, vocab, maxlen)

def strs2seq(sentences:str, vocab, tokenizer, maxlen:int):
    X = []
    # print(len(sentences))
    # print(type(sentences))
    for sentence in sentences:
        X.append(str2seq(sentence, vocab, tokenizer, maxlen))
    return torch.stack(X)

#####  攻击中要用到的函数  ######
def get_random(s:int, e:int, weights=None):
    if weights is not None:
        return random.choices([i for i in range(s, e+1)], weights)[0]
    return random.randint(s, e)   

def read_standard_txt_data(path):
            # 处理的数据格式为(label,sentence) 返回data列表和label列表
        data = []
        labels = []
        with open(path, 'r', encoding='utf-8') as file:
           total_lines = sum(1 for _ in file)
            # 重新将文件指针移回文件开头
           file.seek(0)
            # 使用tqdm包装文件对象
           progress_bar = tqdm(file, total=total_lines, desc='data loading... ')
           for line in progress_bar:
               line = line.strip('\n')
               data.append(line[2:].strip()) # 要求数据处理为 “label”,"data"
               labels.append(int(line[0])) #要求数据处理为 “label”,"data"
        logging.info(f'loading data {len(data)} from {path}')
       
        return data, labels
    
# 将抽取的数据和标签以“datas+label”的形式写入path中
def write_standard_data(datas, labels, path, mod='w'):#data是抽样的数据 labels是抽样数据对应的标签
    assert len(datas) == len(labels)
    num = len(labels)
    logging.info(f'writing standard data {num} to {path}')
    with open(path, mod, newline='', encoding='utf-8') as file:
        for i in range(num):
            file.write(str(labels[i])+','+datas[i]+'\n')