import torch
from torch import nn
from myrelu import GBNR
import numpy as np
from config import args,ballConfig
from torch.nn import functional as F
from tool import find_nearest_center, Vote_Neark_label,Vote_Neark
from dataProcess import Vocab
from dataForBert import Sentence
import random
import os
from collections import deque
from transformers import BertModel,XLNetModel ,AutoTokenizer, AutoModelForMaskedLM

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
seed_torch()

class LSTM(nn.Module):
    
    def __init__(self, word_dim,hidden_size,num_layers, vocab:Vocab, labels_num, dropout,bid=False):
        super().__init__()
        
        self.bid = bid
        self.hidden_size = hidden_size
        self.center_data = torch.empty(0,hidden_size)
        self.center_label = torch.empty(0)
        self.balls = [] 
        
        if self.bid:
            self.model_name = 'BiLSTM'
        else:
            self.model_name = 'LSTM'
            
        self.vocab_size = vocab.num # 词库大小

        self.encoder = nn.LSTM(word_dim,hidden_size = self.hidden_size,num_layers= num_layers,
                               dropout=dropout, bidirectional=bid,batch_first=True) # 定义lstm层
        
        # 定义[vocab_size,word_dim]的嵌入层
        self.embedding = nn.Embedding(num_embeddings= self.vocab_size,embedding_dim=word_dim,_weight=torch.from_numpy(vocab.vectors).float()) 
       
        if args.model_train:
            self.encoder.weight.requires_grad = True
            self.embedding.weight.requires_grad = True # embedding嵌入层的权重也需要更新
        
        # 定义分类器层，输出4分类
        self.classifier = nn.Linear(in_features=self.hidden_size, out_features=labels_num, bias=False)

        
    def predict_class(self, X: torch.Tensor, flag=False ):
        if X.dim() == 1:
            X = X.view(1, -1) 
        predicts = None
        with torch.no_grad():
            logits= self(X,None,flag=2,purity=1) # 获取中间向量 接下来进行投票

            _, predicts =  Vote_Neark_label(logits,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
            predicts = predicts.to(args.device)
        if flag: return logits, predicts
        return predicts
    
    def predict_prob(self, X: torch.Tensor, y_true: torch.Tensor):
        # if self.training:
        #     raise RuntimeError('you shall take the model in eval to get probability!')    0
        if X.dim() == 1:
            X = X.view(1, -1)
        if y_true.dim() == 0:
            y_true = y_true.view(1)
        with torch.no_grad():
            logits= self(X,None,flag=2,purity=1) # 获取中间向量 接下来进行投票
            logits, predicts = Vote_Neark(logits,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
            # print(logits)
            logits = F.softmax(logits, dim=1)
            # print(logits)
            # print(y_true)
            prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
            return prob
    
    def forward(self, x, target_label, flag, purity):
        """
        x:(bs,seq_len)
        target_label: 文本的原标签
        flag: 用于标志训练、防御、是否加粒球等各个情况
        purity:聚球纯度
        """
        ### 以rate比例对x进行掩码（1-30） 
        batch_size,pad_len = x.shape  
        mask_num = int(args.rate*pad_len)                                        
        mask = np.zeros((batch_size, pad_len), dtype=bool)
        for i in range(batch_size):
            mask[i, :mask_num] = True
            np.random.shuffle(mask[i, :])  # 打乱掩码的位置 
    
        x = torch.where(torch.from_numpy(mask).to(args.device), torch.tensor(0).to(args.device), x.to(args.device)) 
        # x = np.where(mask, 0, x)# x = np.where(mask, 0, x.detach().cpu())
        # x = torch.tensor(x).to(device)
      
        embed = self.embedding(x) # 将每个词token_id变成向量 embed.size: (bs,seq_len,word_dim)
        # print(embed.shape)
        out,_ = self.encoder(embed) # 语句向量输入lstm， out: (bs,seq_len,hidden_size)
        # print(out.shape)
        
        if self.bid:
            out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:] 
            # 将正反两个输出对应的位置进行求和。得到的out的形状为[batch_size,seq_len,128]
        
         ### 1.平均池化
        # out = out.mean(dim=1).contiguous() #求out维度1（seq_len）上的均值，并去掉求均值的维度[batch_size,128]
        # out = out.view(out.size(0), -1) # -1表示自动确定维度 ->[batch_size,128] (batch_size,hidden_size)
        
        ### 2.自注意力池化
        attention_weights = F.softmax(out, dim=1) # 计算注意力权重
        out = torch.sum(attention_weights * out, dim=1) # 使用注意力权重计算加权平均 (batch_size, hidden_size)
       
        out = out.view(out.size(0), -1) # (batch_size,hidden_size)
        
        #### (1)训练阶段： 过粒球层 粒球聚类也需要被优化
        if flag == 0: 
            ### 把输入数据，调成粒球输入的形式
            data = out
            origin_label = target_label
    
            #将原来的标签[0,1,...] 转置成列向量再与out拼接起来 out[batch_size,embedding_dim+1]
            out = torch.cat((target_label.reshape(-1, 1), out), dim=1)  

            pur_tensor = torch.Tensor([[purity]] * out.size(0)) # 为每个batch_size复制一个纯度
            out = torch.cat((pur_tensor.to(args.device), out), dim=1).to(args.device) #out (batch_size,purity+target+features)
			# 输入粒球层，聚球，输出 out,是球心向量，target：球心标签  balls:球中的数据（LLP：将返回的balls删除了）
            out, target_label= GBNR.apply(out.to(args.cpu_device))  # apply对张量的每个元素应用to(cup_device)

            # 保存聚好的粒球中心和标签
            self.center_data = out # size: (ball_num,hidden_size)
            self.center_label = target_label # size: (ball_num)
        
            out, target_label= out.to(args.device), target_label.to(args.device)
            out = self.classifier(out) 
            return data.to(args.device), out.to(args.device), origin_label.to(args.device) ,self.center_data.to(args.device),target_label.to(args.device)
      
        ### 没有用到 
        if flag == 1:
           ### 没有用到 把输入的测试数据，调成粒球输入的形式
            
            data = out # 原数据备份
            
            #将原来的标签[0,1,...] 转置成列向量再与out拼接起来 out[batch_size,embedding_dim+1]
            out = torch.cat((target_label.reshape(-1, 1), out), dim=1)  
            pur_tensor = torch.Tensor([[purity]] * out.size(0)) # 为每个batch_size复制一个纯度
            out = torch.cat((pur_tensor.to(args.device), out), dim=1)  #out (batch_size,purity+target+features)
			
			# 输入粒球层，聚球，输出 out,是球心向量，target：球心标签  balls:球中的数据（LLP：将返回的balls删除了）
            out, target_label= (GBNR.apply(out.to(args.cpu_device)) ).to(args.device) # apply对张量的每个元素应用to(cpu_device)
            # 保存聚好的粒球中心和标签
            self.center_data = out # size: (ball_num,hidden_size)
            self.center_label = target_label # size: (ball_num)
            
            out,indices = find_nearest_center(data, self.center_data)  # 找到原样本与测试数据聚的球心最近的球心和下标
            out= out.to(args.device)
            
            # 找出对应的标签，此时的形状是[tensor(label1),tensor()...]
            target_label = [self.center_label[i] for i in indices] 
            
            # 将label转换成tensor([label1...])
            target_label = torch.stack(target_label).to(args.device) 
            out = self.classifier(out)  # 将球心向量输入分类器得到概率 out: ([ball_num, label_num])
        
            return out.to(args.device), target_label.to(args.device) ,self.center_data.to(args.device),self.center_label.to(args.device)
        
       
        #### (3)在训练不过粒球模型，直接用LSTM+classifier进行预测  
        if flag == -1:
            out = self.classifier(out)
            return out
        
        #### (4)直接使用LSTM出来的logit不经过分类器与粒球 
        if flag == 2:
            return out  
        
class TextCNN(nn.Module):
    def __init__(self,word_dim,vocab:Vocab,labels_num,kernel_sizes,
                 num_channels, dropout):
        super().__init__()
        self.vocab_size = vocab.num
        self.center_data = torch.empty(0,word_dim)
        self.center_label = torch.empty(0)
        # 模型架构定义
        # self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=word_dim)
        self.embedding = nn.Embedding(num_embeddings= self.vocab_size,embedding_dim=word_dim,_weight=torch.from_numpy(vocab.vectors).float())
        self.embedding.weight.requires_grad = True 
        
        #定义池化层，同时 仅仅针对AdaptiveMaxPool1d关闭 cudnn.deterministic， 
        
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.ModuleList() # 卷积层列表
        for c, k in zip(num_channels, kernel_sizes):
            self.encoder.append(
                nn.Conv1d(in_channels=word_dim,
                          out_channels=c,
                          kernel_size=k)
            )
       
        self.fc = nn.Linear(sum(num_channels), word_dim)
        self.classifier = nn.Linear(in_features=word_dim, out_features=labels_num)
        
    def forward(self,x,target_label,flag,purity):
        x = x.to(args.device)
        emd = self.embedding(x) # (bs,seq_len,word_dim)
        
        embeddings = self.dropout(emd)
        embeddings = embeddings.permute(0, 2, 1) # [bs, word_dim, seq_len]
        
        outs = torch.cat(
            [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.encoder], dim=1)
        outs = self.fc(outs)
        out = self.dropout(outs) # (128,150) [bs,sum(channel_size)]
       
        
        # attention_weights = F.softmax(emd, dim=1) # 计算注意力权重
        # out = torch.sum(attention_weights * outs, dim=1) # 使用注意力权重计算加权平均 (batch_size, hidden_size)
        # out = out.view(out.size(0), -1) # -1表示自动确定维度 ->[batch_size,128] (batch_size,hidden_dim) 
        
        #### (1)训练阶段： 过粒球层 粒球聚类也需要被优化
        if flag == 0: 
            ### 把输入数据，调成粒球输入的形式
            data = out
            origin_label = target_label
            #将原来的标签[0,1,...] 转置成列向量再与out拼接起来 out[batch_size,embedding_dim+1]
            out = torch.cat((target_label.reshape(-1, 1), out), dim=1)  
            pur_tensor = torch.Tensor([[purity]] * out.size(0)) # 为每个batch_size复制一个纯度
            out = torch.cat((pur_tensor.to(args.device), out), dim=1)  #out (batch_size,purity+target+features)
            
            # 输入粒球层，聚球，输出 out,是球心向量，target：球心标签  balls:球中的数据（LLP：将返回的balls删除了）
            out, target_label= GBNR.apply(out.to(args.cpu_device))  # apply对张量的每个元素应用to(cup_device)
            # out, target_label= GBNR.apply(out.to(args.device)) 
            # 保存聚好的粒球中心和标签
            self.center_data = out # size: (ball_num,hidden_size)
            self.center_label = target_label # size: (ball_num)
        
            out, target_label= out.to(args.device), target_label.to(args.device)
            out = self.classifier(out) 
            return data, out, origin_label ,self.center_data,target_label
        
        #### (2)训练中的评估阶段： 数据经过lstm 再找到对应的训练好的粒球中心
        if flag == 1:
            data = out # 原数据备份
            
            #将原来的标签[0,1,...] 转置成列向量再与out拼接起来 out[batch_size,embedding_dim+1]
            out = torch.cat((target_label.reshape(-1, 1), out), dim=1)  
            pur_tensor = torch.Tensor([[purity]] * out.size(0)) # 为每个batch_size复制一个纯度
            out = torch.cat((pur_tensor.to(args.device), out), dim=1)  #out (batch_size,purity+target+features)
			
			# 输入粒球层，聚球，输出 out,是球心向量，target：球心标签  balls:球中的数据（LLP：将返回的balls删除了）
            out, target_label= GBNR.apply(out.to(args.cpu_device))  # apply对张量的每个元素应用to(cup_device)
            # 保存聚好的粒球中心和标签
            self.center_data = out # size: (ball_num,hidden_size)
            self.center_label = target_label # size: (ball_num)
            
            out,indices = find_nearest_center(data, self.center_data) 
            out= out.to(args.device)
            
            # 找出对应的标签，此时的形状是[tensor(label1),tensor()...]
            target_label = [self.center_label[i] for i in indices] 
            
            # 将label转换成tensor([label1...])
            target_label = torch.stack(target_label).to(args.device) 
            out = self.classifier(out)  # out: ([ball_num, label_num])
           
            return out, target_label ,self.center_data,target_label
    
        #### (3)在训练不过粒球模型，直接用LSTM+classifier进行预测  
        if flag == -1:
            out = self.classifier(out)
            
            return out
        
        #### (4)在apply阶段，直接使用LSTM出来的logit不经过分类器与粒球 
        if flag == 2:
            return out  
        
    def predict_class(self, X: torch.Tensor, flag=False ):
        if X.dim() == 1:
            X = X.view(1, -1)
        predicts = None
        with torch.no_grad():
            logits= self(X,None,flag=2,purity=1) # 获取中间向量 接下来进行投票
            _, predicts = Vote_Neark_label(logits,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
        if flag: return logits, predicts
        predicts = predicts.to(args.device)
        return predicts
    
    def predict_prob(self, X: torch.Tensor, y_true: torch.Tensor):
        if X.dim() == 1:
            X = X.view(1, -1)
        if y_true.dim() == 0:
            y_true = y_true.view(1)
        with torch.no_grad():
            logits= self(X,None,flag=2,purity=1) # 获取中间向量 接下来进行投票
            logits, predicts = Vote_Neark_label(logits,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
            logits = F.softmax(logits, dim=1)
            prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
            return prob



MODEL_PATHS = {
    'Bert': './bert',
    'RoBERTa': './roberta',
    'XLNet': './xlnet'
}
    
class myModel(nn.Module):
    
    def __init__(self, hidden_size,word_dim,labels_num, dropout):
        super().__init__()

        self.model_path = MODEL_PATHS[args.model]
        self.hidden_size = hidden_size
        self.word_dim = word_dim
        self.center_data = torch.empty(0,hidden_size)
        self.center_label = torch.empty(0)
        self.balls = []
        
        if args.model=='Bert':
             self.myModel =BertModel.from_pretrained(self.model_path,output_hidden_states=True).to(args.device)
             # output_hidden_states=True 确保加载模型时返回所有隐藏状态，而不仅仅是最终隐藏状态。
        if args.model == 'RoBERTa':
            torch.set_default_tensor_type(torch.FloatTensor)
            self.myModel = AutoModelForMaskedLM.from_pretrained(self.model_path,output_hidden_states=True).to(args.device)
        if args.model == 'XLNet':
            self.myModel = XLNetModel.from_pretrained(self.model_path,output_hidden_states=True).to(args.device)
            
        self.encoder = nn.Linear(in_features=self.hidden_size, out_features=self.word_dim).to(args.device)
        
        self.classifier = nn.Linear(in_features=self.word_dim, out_features=labels_num, bias=False).to(args.device)
        self.dropout = nn.Dropout(p=dropout)
    

    def predict_class(self, X: torch.Tensor, flag=False ):
        predicts = None
        processor = Sentence(args.dataset)
        sen = processor.process(X)
        with torch.no_grad():
            logits= self(sen,None,flag=2,purity=1) # 获取中间向量 接下来进行投票

            _, predicts = Vote_Neark_label(logits,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
            predicts = predicts.to(args.device)
        if flag: return logits, predicts
        return predicts
    
    def predict_prob(self, X: torch.Tensor, y_true: torch.Tensor,flag=False):
        # if self.training:
        #     raise RuntimeError('you shall take the model in eval to get probability!')
        if y_true and y_true.dim() == 0:
            y_true = y_true.view(1)
        processor = Sentence(args.dataset)
        sen = processor.process(X)
        with torch.no_grad():
            logits= self(sen,None,flag=2,purity=1) # 获取中间向量 接下来进行投票
            logits, predicts = Vote_Neark_label(logits,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
           
            logits = F.softmax(logits, dim=1)
           
            prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
           
        # if flag:
        #     return logits   
        return torch.tensor(prob)
        
    def forward(self, x,attention_mask,token_type_ids, target_label, flag, purity):
        """        adv_y = model.predict_prob(perturbed_vector, true_y)

        x:(bs,seq_len)
        target_label: 文本的原标签
        flag: 用于标志训练、防御、是否加粒球等各个情况
        purity:聚球纯度
        """  
        
        if args.model=='RoBERTa':
            # print(x['input_ids'].shape) # batch_size,padding_len
            pooler_output = self.myModel(x.to(args.device), attention_mask = attention_mask.to(args.device))
            # pooler_output是一个长度为2的元组，第一个元素的大小为(bs,padding_len,50265)表示对每个位置都输出一个词库中的概率
            # 第二个元素为每个隐藏层的状态
            out = pooler_output[1][len(pooler_output[1])-1]
            # print(out.shape) # bs,seq_len,hidden_size）
        else:
            pooler_output = self.myModel(x.to(args.device), attention_mask = attention_mask.to(args.device),
                            token_type_ids = token_type_ids.to(args.device)) # 将每个词token_id变成向量 embed.size: (bs,seq_len,768)

            out = pooler_output.last_hidden_state ## （bs,seq_len,hidden_size）
        #print("out before mean",out.shape)
         ### 1.平均池化
        out = out.mean(dim=1).contiguous() #求out维度1（seq_len）上的均值，并去掉求均值的维度[batch_size,hidden_dim]
        out = out.view(out.size(0), -1) # -1表示自动确定维度 ->[batch_size,128] (batch_size,hidden_size)
        
        ### 2.自注意力池化
        # attention_weights = F.softmax(out, dim=1) # 计算注意力权重
        # out = torch.sum(attention_weights * out, dim=1) # 使用注意力权重计算加权平均 (batch_size, hidden_size)
       
        #out = out.view(out.size(0), -1) # (batch_size,hidden_size)

        out= (self.encoder(out)).to(args.device) # out: (bs,hidden_size)

        # out = self.dropout(out)
        #### (1)训练阶段： 过粒球层 粒球聚类也需要被优化
        if flag == 0:
            # 调整输入数据形状
            data = out.clone()
            origin_label = target_label.clone()
           # print("out",out.shape)
           # print("target",target_label.shape)
            # 将标签与输出拼接
            # print("origin_lable:",origin_label.shape)
            # print("data:", data.shape)

            out = torch.cat((target_label.reshape(-1, 1), out), dim=1)
            
            # 为每个样本复制一个纯度
            pur_tensor = torch.full((out.size(0), 1), purity)
            out = torch.cat((pur_tensor.to(args.device), out), dim=1)
            
            # 输入粒球层，聚球
            out, target_label = GBNR.apply(out.to(args.cpu_device))
            
            # 保存聚好的粒球中心和标签
            self.center_data =  out.to(args.device)
            self.center_label = target_label.to(args.device)
            
            # 将粒球中心过分类器得到分类后的logit
            out = self.classifier(self.center_data)
            
            return data, out, origin_label, self.center_data, self.center_label
        
        #### 用于textattack攻击测试,返回的是预测的分数
        if flag == 1:
            data = out # 原数据备份
            logits, predicts = Vote_Neark_label(data,self.classifier,ballConfig.loaded_ball_centers, ballConfig.loaded_center_labels, args.k)
            logits = F.softmax(logits, dim=1)
            
            return logits
       
        #### (3)在训练不过粒球模型，直接用LSTM+classifier进行预测  
        if flag == -1:
            data = out
            out = self.classifier(out)
            return data, out
        
        #### (4)直接使用LSTM出来的logit不经过分类器与粒球 
        if flag == 2:
            return out  
        
class GBQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)

    def push(self,ball_data):
        for row in ball_data:
            self.queue.append(row)

    def pop_if_full(self):
        if len(self.queue) == self.queue.maxlen:
            self.queue.popleft()
            
    def concatTensors(self):
        if len(self.queue) > 0:
            concat_tensor = torch.stack(list(self.queue), dim=0)
            return concat_tensor
        else:
            return None       
    def get_len(self):
        return len(self.queue)
    
    def get_queue(self):
        return list(self.queue)