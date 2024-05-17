import os
import numpy as np
import random
import torch
import logging
from LM import LSTM,TextCNN
from dataProcess import MyDataset,Tokenizer,Vocab, Sentence
from torch.utils.data import DataLoader
import torch.nn as nn

from config import args,config_data,LSTMConfig, TextCNNConfig
from Attacker import adversarial_paraphrase,\
        textfool_perturb_text, random_attack, GAAdversary
from tool import str2seq,read_standard_txt_data,write_standard_data

"""
攻击步骤：
 1.初始化模型，将训练好的模型参数加载到初始化的模型框架中
 2.从测试数据中抽样 进行攻击测试
 
 未完成：抽样、调用函数进行攻击
"""                   
#设置随机种子，以确保实验的可复现性
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)  
seed_torch()

## 参数设置    
dataset_name = args.dataset
device = args.device
model_name = args.model
pretrained_model_path = args.pretrained_model_path

attack_method = args.attack_method
sub_rate_limit = args.sub_rate_limit
## 攻击后数据存储的位置  (动态攻击，攻击成功就储存)
output_dir = f'./data_attacked/{args.dataset}/{attack_method}/{model_name}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

adv_path = output_dir + '_adv.txt'

if device==torch.device('cuda'):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' 
    
maxlen = config_data[dataset_name].padding_maxlen
label_num = config_data[dataset_name].labels_num # 对应数据集的标签数


single_label_num = int(1000/label_num) # 单类标签抽样条数
## 抽样的数据存储的位置
samples_path = config_data[dataset_name].data_dir + '/clean_samples.txt'
if os.path.exists(samples_path):
    clean_path = samples_path
else:
    clean_path = None
    
## 创建文件储存训练信息
# （1）创建log文件夹
if not os.path.exists('logs'):
    os.makedirs('logs')
# （2）设置日志记录器
log_file = f'./logs/Fool_{model_name}_{dataset_name}_{attack_method}_info.log' # 日志文件名称
# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
fh = logging.FileHandler(log_file,encoding="utf-8",mode="a")
formatter = logging.Formatter("%(asctime)s - %(name)s-%(levelname)s %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

## 读取训练数据，创建词库  
tokenizer = Tokenizer()
train_data = MyDataset(dataset_name, tokenizer, is_train=True, data_path=config_data[dataset_name].train_data_path)
test_data = MyDataset(dataset_name, tokenizer, is_train=False, data_path=config_data[dataset_name].test_data_path)
vocab = Vocab(train_data.data_token, vocab_limit_size=config_data[dataset_name].vocab_limit_size,glove_path = './dataset/glove.840B.300d.txt') # 用训练样本构成词库
train_data.token2seq(vocab, maxlen) # 对训练样本进行编码得到[index1,index2......],[index......]
test_data.token2seq(vocab, maxlen)

class Attack():
    def __init__(self,sample_size_single_label,clean_samples_path):
         ## 抽取样本 
        if clean_samples_path is not None:
            self.datas, self.labels = read_standard_txt_data(clean_samples_path)        
        else:
            self.datas, self.labels = test_data.sample_by_labels(sample_size_single_label) # 抽样并写入csv文件,这边直接用测试集抽样
            write_standard_data(self.datas, self.labels, samples_path)
            self.datas, self.labels = read_standard_txt_data(samples_path)
            
    def apply(self, adv_method, Model, tokenizer, vocab, sub_rate_limit):
        
        self.adv_methods = {
            'PWWS': self.get_fool_sentence_pwws,
            'TextFooler': self.get_fool_sentence_textfool,
            'RES': self.get_fool_sentence_random,
            'GA':self.get_fool_sentence_ga,
        }
        assert adv_method in self.adv_methods
        self.adv_methods = self.adv_methods[adv_method]
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.defendModel =Model
        
        self.sub_rate_limit = sub_rate_limit
        self.adv_datas = [] # 保存对抗样本
        self.verbose = False
        self.k = args.k # 投票机制k值
        success_num = 0 # 攻击成功的条数记录 
        failure_num = 0 # 攻击失败的次数记录
        try_all = 0 # 总共尝试攻击的数据
        pre_num = 0 # 模型预测正确的次数
        
        
      
        for idx, data in enumerate(self.datas): # 遍历抽样数据的每条样本
                logger.info('\n')
                logger.info("**********************************%dth-sample************************************",idx)
                label = self.labels[idx]
                adv_s, flag, pre_num = self.adv_methods(data, self.labels[idx], idx, pre_num) # 返回攻击后的语句、攻击成功标志、预测正确的条数
                self.adv_datas.append(str(adv_s)) # 将攻击后的样本添加到adv_datas列表
                # 如果攻击成功
                if flag==1:         
                    success_num += 1
                    try_all += 1      
                    logger.info('The %dth adv successfully crafted, success rate is  %.5f',idx,success_num/try_all)
                    
                elif flag==0:
                    failure_num += 1
                    try_all += 1
                    logger.info("The %dth adv example failed crafted, fail rate is %.5f",idx,failure_num/try_all)
                
                elif flag==-1: # 预测失败 不攻击
                    try_all += 1
        
        with open(adv_path, 'w', newline='',encoding='utf-8') as file: # 将对抗样本保存到txt文件
        
            for i in range(len(self.adv_datas)):
                file.write(self.adv_datas[i] + '\n')
        
        logger.info("预测正确次数为：%d, 攻击成功次数：%d 攻击失败次数：%d 攻击后模型的准确率为: %.5f",pre_num,success_num,failure_num,(pre_num-success_num)/1000)

    def get_fool_sentence_pwws(self, sentence:str, label:int, index:int, pre_num):
        
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(device) # 将单句样本转换成向量(只是tokens) [word_dim]
        label = torch.tensor(label).to(device) # 将label移动到指定设备
        flag = -1 # 用于记录攻击是否成功
        predict = Model.predict_class(vector).to(device)
    
        if predict == label:
            pre_num+=1
            logger.info("模型预测正确，目前模型的预测正确数为：%d,开始进行攻击", pre_num)
            sentence, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(
                sentence, vector, label, self.tokenizer, self.vocab, self.defendModel, self.verbose,
                self.sub_rate_limit)
            if adv_y != label: # 攻击成功
                flag = 1
                logger.info("本次攻击成功，替换率为：{} 命名实体替换率为：{} 扰动细节：{}".format(sub_rate,NE_rate,change_tuple_list))
            else:
                flag = 0
                logger.info("本次攻击失败")
        else:
            logger.info("模型对本次样本预测失败~~")
        return sentence, flag, pre_num    
    
    def get_fool_sentence_textfool(self, sentence:str, label:int, index:int, pre_num):
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(device) # 将单句样本转换成向量(只是tokens) [word_dim]
        label = torch.tensor(label).to(device) # 将label移动到指定设备
        flag = -1 # 用于记录攻击是否成功
        predict = Model.predict_class(vector).to(device)
  
        if predict==label:
            pre_num+=1
            logger.info("模型预测正确，目前模型的预测正确数为：%d,开始进行攻击", pre_num)
            sentence, adv_y, sub_rate, NE_rate, change_tuple_list = textfool_perturb_text(
                sentence, self.defendModel, self.vocab, self.tokenizer, maxlen, label, False,
                verbose=self.verbose, sub_rate_limit=self.sub_rate_limit,
            )
            if adv_y != label: # 攻击成功
                flag = 1
                logger.info("本次攻击成功，替换率为：{} 命名实体替换率为：{} 扰动细节：{}".format(sub_rate,NE_rate,change_tuple_list))
            else:
                flag = 0
                logger.info("本次攻击失败")
        else:
            logger.info("模型对本次样本预测失败~~")
        return sentence, flag, pre_num 
    
    def get_fool_sentence_random(self, sentence:str, label:int, index:int, pre_num):
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(device) # 将单句样本转换成向量(只是tokens) [word_dim]
        label = torch.tensor(label).to(device) # 将label移动到指定设备
        flag = -1 # 用于记录攻击是否成功
        predict = Model.predict_class(vector).to(device)
        if predict==label:
            pre_num+=1
            logger.info("模型预测正确，目前模型的预测正确数为：%d,开始进行攻击", pre_num)
            sentence, adv_y, sub_rate, NE_rate, change_tuple_list =  random_attack(
                sentence,label,self.defendModel, self.vocab, self.tokenizer, maxlen, 
                verbose=self.verbose, sub_rate_limit=self.sub_rate_limit,
            )
            if adv_y != label: # 攻击成功
                flag = 1
                logger.info("本次攻击成功，替换率为：{} 命名实体替换率为：{} 扰动细节：{}".format(sub_rate,NE_rate,change_tuple_list))
            else:
                flag = 0
                logger.info("本次攻击失败")
        else:
            logger.info("模型对本次样本预测失败~~")
        return sentence, flag, pre_num 
    
    def get_fool_sentence_ga(self, sentence:str, label:int, index:int, pre_num):
        vector = str2seq(sentence, self.vocab, self.tokenizer, maxlen).to(device) # 将单句样本转换成向量(只是tokens) [word_dim]
        label = torch.tensor(label).to(device) # 将label移动到指定设备
        flag = -1 # 用于记录攻击是否成功
        predict = Model.predict_class(vector).to(device)
        if predict==label:
            pre_num+=1
            logger.info("模型预测正确，目前模型的预测正确数为：%d,开始进行攻击", pre_num)
            success, sentence, _ = GAAdversary(self.defendModel, self.vocab, self.tokenizer, maxlen).run(sentence, label)
            if success:
                flag = 1
                logger.info("本次攻击成功")
            else:
                flag = 0
                logger.info("本次攻击失败")
        else:
            logger.info("模型对本次样本预测失败~~")
        return sentence, flag, pre_num         
        
if __name__ == '__main__':      
    #### 初始化目标模型和防御
    if model_name=='LSTM':    
        num_hiddens = LSTMConfig.num_hiddens[dataset_name] # 128
        num_layers = LSTMConfig.num_layers[dataset_name] # 2
        word_dim = LSTMConfig.word_dim[dataset_name] # 300
        Model = LSTM(word_dim,hidden_size=num_hiddens, num_layers=num_layers,vocab=vocab,labels_num = label_num,dropout=0.1,bid=False)
            
    elif model_name=='BidLSTM':
        num_hiddens = LSTMConfig.num_hiddens[dataset_name]
        num_layers = LSTMConfig.num_layers[dataset_name]   
        word_dim = LSTMConfig.word_dim[dataset_name]
        Model =LSTM(word_dim,hidden_size=num_hiddens,num_layers=num_layers,vocab=vocab,labels_num = label_num,dropout=0.1,bid=True)
        
    elif model_name=='TextCNN':
        word_dim = TextCNNConfig.word_dim[dataset_name]
        kernel_sizes = TextCNNConfig.channel_kernel_size[dataset_name][1]
        num_channels = TextCNNConfig.channel_kernel_size[dataset_name][0]
        Model = TextCNN(word_dim, vocab,label_num, kernel_sizes, num_channels, dropout=0.1)

        ##加载防御模型和目标模型的权重


    if(torch.cuda.is_available()):
        Model_weight = torch.load(pretrained_model_path) # 如果显卡可用 将模型加载到其训练时的存储设备中 gpu或者cpu
    else:
        Model_weight = torch.load(pretrained_model_path,'cpu') # 显卡不可用则加载到cpu上

    Model.load_state_dict(Model_weight['model_state_dict'])
    Model.to(device)
    Model.eval()
    classifer = Model.classifier #单独提取出classifier的权重
    
    ## 实例化Attack类，进行攻击
    attacker = Attack(single_label_num, clean_path)
    attacker.apply(attack_method,Model,tokenizer,vocab,sub_rate_limit)