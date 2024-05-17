import torch
import logging
from transformers import  BertTokenizer,AutoTokenizer,XLNetTokenizer, XLNetModel,AutoModelForMaskedLM

from torch.utils.data import Dataset
from tqdm import tqdm
from config import *

import random

# 默认分词器
BERT_PATH = './bert'
Tokenizer = BertTokenizer.from_pretrained(BERT_PATH) 

if args.model=='RoBERTa':
    MODEL_PATH = './roberta'
    Tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
elif args.model=='XLNet':
    MODEL_PATH = './xlnet'
    Tokenizer = XLNetTokenizer.from_pretrained(MODEL_PATH)    
class MyDataset(Dataset):
    def read_standard_data(self, path):
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
               data.append(line[5:-1].strip()) # 要求数据处理为 “label”,"data"
               labels.append(int(line[1])) #要求数据处理为 “label”,"data"
        logging.info(f'loading data {len(data)} from {path}')
       
        return data, labels
    
    ## 抽样样本
    def read_standard_txt_data(self, path):
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
    
    def data2token(self):
        # self.data的格式是[sentence1,sentence2....]
        logging.info(f'is_train: {self.is_train} , data is to tokens!')
        assert self.data is not None
        for sen in self.data:
            tokens = ['[CLS]'] + Tokenizer.tokenize(sen) +['[SEP]']
            self.data_token.append(tokens)
    
    def token2seq(self, maxlen:int):
        if len(self.token_ids) > 0:
            self.token_ids.clear()
            # self.labels_tensor.clear()
        logging.info(f'is_train: {self.is_train} , data is to sequence!')
        self.maxlen = maxlen
        assert self.data_token is not None # data_tokens: [[sen1_tok1, sen1_tok2..],[...], [senn_tok1,...]]
        for tokens in self.data_token:
            padded_tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
            token_id = Tokenizer.convert_tokens_to_ids(padded_tokens)
            self.token_ids.append(token_id)
        # for label in self.labels:
        #     self.labels_tensor.append(torch.tensor(label)) # labels_tensor： shape seq_len
    
    def split_data_by_label(self):
        datas = [[] for _ in range(self.labels_num)] # 创建labels_num个空数组，存储在datas列表中。每个列表可以存储单个标签的数据    
        for idx, lab in enumerate(self.labels): # self.labels=>数据集的所有label
            temp = (self.data[idx], lab) # 定义了一个元组，包含数据第idx和标签
            datas[lab].append(temp) # 将temp添加到对应的datas列表中
        return datas
    
    def sample_by_labels(self, single_label_num:int):
        random.seed(42)  ### 改随机种子 抽取不同的样本做测试
        datas = self.split_data_by_label() # 按标签分数据  得到的是一个列表
        sample_data = []
        sample_label = [-1 for _ in range(single_label_num*self.labels_num)]# 创建一个single_label_num*self.labels_num的数组，每个元素初始值设为-1.其实数组大小就是随机抽取的样本数
        for i in range(self.labels_num):
            sample_data += random.sample(datas[i], single_label_num) # 从每个标签数据中，各取single_label_num条数据
        for idx, data in enumerate(sample_data):
            sample_data[idx] = data[0]
            sample_label[idx] = data[1]
        assert len(sample_data) == len(sample_label)
        return sample_data, sample_label
    
    def __len__(self):
        
        return len(self.labels_tensor)

    def __getitem__(self, item):
        return (self.data[item], self.labels_tensor[item])
        # inputs = Tokenizer(self.data[item], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        # return inputs, self.labels_tensor[item]
    
    def __init__(self, dataset_name,is_train:bool,data_path, is_to_tokens=True):
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.labels_num = config_data[dataset_name].labels_num
        self.data = [] # 用于接收数据文件中的sentence
        self.labels = [] # 用于记录数据集中的labels
        self.data_token = [] # 用于记录每条样本分词加[CLS],[SEP] ["[CLS]",token1,token2,..."[SEP]"]
        self.token_ids = []
        self.labels_tensor = []
        self.tokenizer = Tokenizer
        self.max_len = config_data[dataset_name].padding_maxlen
        
        if isinstance(data_path, str):
                data_path = [data_path] # 如果路径是字符串类型，将路径转换成列表格式
        
        for path in data_path:
            if path.lower().endswith('.csv'):    
                td, tl = self.read_standard_data(path) # 返回的是数据列表和label列表
            if  path.lower().endswith('.txt'):
                td, tl = self.read_standard_txt_data(path)
            self.data += td # 将数据合并到self.data列表中
            self.labels += tl # 将label合并到self.labels列表中
        
        for label in self.labels:
                 self.labels_tensor.append(torch.tensor(label))
                 
        if is_to_tokens:
            self.data2token()

#  对单句数据进行处理的类   
class Sentence():
   
    def __init__(self, dataset_name):
           self.dataset_name = dataset_name
           self.labels_num = config_data[dataset_name].labels_num
           self.max_len = config_data[dataset_name].padding_maxlen
           self.tokenizer = Tokenizer   
           self.data_token = []
           self.token_id =[]
        #    self.token_id = torch.tensor([])

    """
    1. 语句分词并加上分隔符
    2. 处理成保留长度，填充
    3. 使用bert内置token编码
    """
    def process(self,sentence):
        # tokens = self.data2token(sentence)
        # seq =  self.__encode_tokens(tokens)
        feature= Tokenizer(sentence, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return feature
    
    
## Bert系列数据处理
# 默认分词器
# BERT_PATH = './bert'
# TokenizerForBert = BertTokenizer.from_pretrained(BERT_PATH) 

# if args.model=='RoBERTa':
#     MODEL_PATH = './roberta'
#     TokenizerForBert= AutoTokenizer.from_pretrained(MODEL_PATH)
        
# elif args.model=='XLNet':
#     MODEL_PATH = './xlnet'
#     TokenizerForBert = XLNetTokenizer.from_pretrained(MODEL_PATH)  
    
# sen = ['I am a bad person , and I am really happy to get a good score ,i dont think it is convert...']

# def seq2vector(data,maxlen):
#     # 将语句分词成token,填充,加[CLS]和[SEP]
#     assert data is not None
#     data_token = []
#     datafeature = []
#     for sen in data:
#        tokens = TokenizerForBert.tokenize(sen) 
#        padded_tokens = ['[CLS]'] + tokens + ['[PAD]' for _ in range(maxlen - len(tokens))] +['[SEP]']
#        datafeature.append(TokenizerForBert.convert_tokens_to_ids(padded_tokens))
#     print(datafeature)
#     feature = TokenizerForBert(data, max_length=maxlen, padding='max_length', truncation=True, return_tensors='pt')
#     print(feature)
#     return datafeature
  
# seq2vector(sen,50)