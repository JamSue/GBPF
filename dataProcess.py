import torch
from nltk.corpus import stopwords
from config import config_data,args
import spacy
import re
import logging
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm import tqdm
from transformers import  BertTokenizer,AutoTokenizer,XLNetTokenizer
from textattack.augmentation import EasyDataAugmenter
import csv
import ast
"""
Tokenizer:分词器
MyDataset: 读取数据集，将数据集处理成 self.data_token:[token1,token2....]， 也可以通过vocab对token进行编码
Vocab:对相应数据集建立词库
Setence: 对单语句进行处理
"""


class Tokenizer():
    
    def __init__(self, tokenizer_type='normal', remove_stop_words=False):
        self.is_remove_stop_words = remove_stop_words
        if tokenizer_type == 'normal':
            self.tokenizer = self.normal_token
        elif tokenizer_type == 'spacy':
            self.nlp = spacy.load('en_core_web_sm')
            self.tokenizer = self.spacy_token
        else:
            raise RuntimeError(f'Tokenizer type is error, do not have type {tokenizer_type}')
        self.token_type = tokenizer_type
        self.stop_words = set(stopwords.words('english')) # 使用nltk的stopwords包创建停用词集合
        for w in ["<br />", '!', ',', '.', '?', '-s', '-ly', '</s>', 's', '</', '>', '/>', 'br', '<']:
            self.stop_words.add(w) # 将上述符号加入停用词集合中
        logging.info(f'using tokenizer {tokenizer_type}, is_remove_stop_words={remove_stop_words}')

    
    def pre_process(self, text: str):
        text = text.lower().strip() # 文本小写化
        text = re.sub(r"<br />", "", text) # 用空字符串替换文本中的换行符
        text = re.sub(r'(\W)(?=\1)', '', text) # 去除重复标点
        text = re.sub(r"([.!?,])", r" \1", text) # 在标点符号前添加空格，以便于将标点符号作为独立的词进行处理
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text) # 替换非字母和". ! ?"的符号       # 返回处理首尾空白字符的文本
        return text.strip() 
    
    def normal_token(self, text: str, is_word=True):
        # 普通分词
        if is_word:
            return [tok for tok in text.split() if not tok.isspace()] 
            # 列表推导式，将text按照空格切分，如果tok不是空白则存放在列表中 返回tok列表
        else:
            return [tok for tok in text] # 将输入的文本分割为单个字符，并以列表的形式返回


    def spacy_token(self, text: str, is_word=True):
        if is_word:
            text = self.nlp(text) # 这行代码使用Spacy库的 nlp 方法对输入的文本进行处理，
                                  # 得到一个Spacy的Doc对象，其中包含了对文本进行了分词、词性标注、实体识别等处理。
            return [token.text for token in text if not token.text.isspace()]
        else:
            return [tok for tok in text]


    def stop_words_filter(self, words: str):
        # 过滤停用词
        return [word for word in words if word not in self.stop_words]


    def __call__(self, text: str, is_word=True):
        # 特殊方式 用于使类的实例像函数一样可调用 => tokenizer(text)  将参数传入该函数进行操作
        text = self.pre_process(text)
        words = self.tokenizer(text, is_word=is_word)
        if self.is_remove_stop_words:
            return self.stop_words_filter(words)
        return words 
    

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
            self.data_token.append(self.tokenizer(sen)) # data_token: [tok1,tok2....]


    def token2seq(self, vocab, maxlen:int):
        if len(self.data_seq) > 0:
            self.data_seq.clear()
            self.labels_tensor.clear()
        logging.info(f'is_train: {self.is_train} , data is to sequence!')
        self.vocab = vocab
        self.maxlen = maxlen
        assert self.data_token is not None # data_tokens: [[sen1_tok1, sen1_tok2..],[...], [senn_tok1,...]]
        for tokens in self.data_token: # 循环对每个语句的tok列表编码
            self.data_seq.append(self.__encode_tokens(tokens)) # 对数据的token列表编码并添加到data_seq列表中=> data_seq:(tensor)[[index1,index2...],....[index...]]  shape:(seq_len,maxlen)
        for label in self.labels:
            self.labels_tensor.append(torch.tensor(label)) # labels_tensor： shape seq_len

    def __encode_tokens(self, tokens):
        '''
        if one sentence length is shorter than maxlen, it will use pad word for padding to maxlen
        :param tokens:
        :return:
        '''
        """对长度不够的sen进行填充(pad:0), 保留长度：maxlen"""

        pad_word = 0
        x = [pad_word for _ in range(self.maxlen)] # 声明应该maxlen长度的0列表
        temp = tokens[:self.maxlen] # 保留maxlen长度后的语句->temp
        for idx, word in enumerate(temp):
            x[idx] = self.vocab.get_index(word) # 从词库中获取token对应的index,将[tok1,tok2..]转换成[index1,index2...]
        return torch.tensor(x)

    def split_data_by_label(self):
        datas = [[] for _ in range(self.labels_num)] # 创建labels_num个空数组，存储在datas列表中。每个列表可以存储单个标签的数据    
        for idx, lab in enumerate(self.labels): # self.labels=>数据集的所有label
            temp = (self.data[idx], lab) # 定义了一个元组，包含数据第idx和标签
            datas[lab].append(temp) # 将temp添加到对应的datas列表中
        return datas
    
    # 抽样，每个标签的数据都抽取1000/label_num条，返回的是数据列表以及标签列表
    # single_label_num: 单个标签抽取的数量(=1000/label_num)    
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
    
    #样本增强并将训练数据集的增强样本添加到csv文件
    def sample_augment(self,num_augmentations):
        augment_data_path =   config_data[self.dataset_name].augment_data_path
        assert self.data is not None
        self.augment_dict.clear() # 清空字典
        augmenter = EasyDataAugmenter(transformations_per_example=num_augmentations)
      
        # for sen in self.data:
        #     aug_samples = augmenter.augment(sen)
        #     self.augment_dict[sen] = set(aug_samples) # 增强样本数组转成set,无序.添加到字典
        # with open(augment_data_path, 'w', newline='') as file:
        #     writer = csv.writer(file)
            
        #     for sen,aug_samples in self.augment_dict.items():
        #         writer.writerow([sen,aug_samples])

        with open(augment_data_path, 'a', newline='') as file:  # 打开文件以追加模式
            writer = csv.writer(file)
            for sen in self.data: 
                aug_samples = augmenter.augment(sen)
                self.augment_dict[sen] = set(aug_samples)  # 增强样本数组转成set,无序.添加到字典
                writer.writerow([sen, self.augment_dict[sen] ]) 
             

    def __len__(self):
            return len(self.labels_tensor)

    def __getitem__(self, item):
        
        # 或者文本，编码和标签都一起返回
        if len(self.data_seq)==0:  
            return (self.data[item], self.labels_tensor[item])
           
        else:
           return (self.data_seq[item], self.labels_tensor[item])
       

    def __init__(self,dataset_name, tokenizer, is_train:bool, data_path, is_to_tokens=True):
           self.is_train = is_train
           self.dataset_name = dataset_name
           self.labels_num = config_data[dataset_name].labels_num
           self.data = [] # 用于接收数据文件中的sentence
           self.labels = [] # 用于记录数据集中的labels
           self.data_token = [] # 用于记录分词后的数据
           self.data_seq = [] # 记录编码后的数据
           self.labels_tensor = []
           self.vocab = None
           self.tokenizer = tokenizer if tokenizer else Tokenizer('normal', remove_stop_words=False)
           self.maxlen = None
           self.augment_dict = {} # 训练数据集的增强字典

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


class Vocab():
       # train_data.data_token为分词后的词列表 [[word11,word12.....],[word21,word22...]....]
    def __init__(self,data_tokens,word_dim:int=300,vocab_limit_size=80000,glove_path = './dataset/glove.840B.300d.txt'):
        """
        data_tokens: shape:(seq_len,word_dim)  [[word11,word12.....],[word21,word22...]....]  
        """
        self.file_path = glove_path
        self.word_dim = word_dim # 一个词的特征向量特征
        self.word_dict = {} # word-index字典
        self.word_count = {} # 记录每个词的词频
        self.vectors = None
        self.num = 0 # word_dict中的词数
        self.data_tokens = []
        self.words_vocab = []
        assert len(data_tokens) > 0
        self.data_tokens = data_tokens # (seq_len,word_dim)
        self.__build_words_index()
        self.__limit_dict_size(vocab_limit_size)
        # if is_using_pretrained:
        #     logging(f'building word vectors from {self.file_path}')
        # 词向量搭建
        self.__read_pretrained_word_vecs()
        logging.info(f'word vectors has been built! dict size is {self.num}')

    
    def __build_words_index(self):
        # 建立词与下标对应的字典
        """
        data_tokens:[sen1,sen2....]  其中sen是由未编码的word（也就是词本身）组成
        word_dict:{word1:index1, word2:index2......}  {词：下标} num表示词库的词数
        word_count:{word1:count1, word2:count2.....} {词：词频} 
    
        """
        for sen in self.data_tokens: # (这边的data_tokens应该是未编码的)
            for word in sen:
                if word not in self.word_dict: # 这边的word_dict应该是对应下标的字典
                    self.word_dict[word] = self.num # 把词添加到字典后面
                    self.word_count[word] = 1
                    self.num += 1
                else:
                    self.word_count[word] += 1

    def __limit_dict_size(self, vocab_limit_size):
        """

        word_count: 这一步中对词频降序排序
        words_vocab: 词频<=limit的词构成的列表 [unk,word1...]
        num: word_vocab的的单词数
        word_dict: 对词频<=limit的词创建的字典  {unk:0, word1,1.....}

        """
        limit = vocab_limit_size
        self.word_count = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True) # 对word_count进行降序排序
        count = 1
        self.words_vocab.append('<unk>') # words_vocab是一个列表，词库中加入'unk'
        temp = {}
       
        for x, y in self.word_count:  # 过滤掉词频高于limit的词
            if count > limit:
                break
            temp[x] = count # temp创建词频<=limit的字典 {word,index} {unk:0,word1:1....}
            self.words_vocab.append(x) # 把词频<=limit的词加入words_vocab   [unk,word1...]
            count += 1 # 记录words_vocab的词数
        self.word_dict = temp
        self.word_dict['<unk>'] = 0
        self.num = count
        assert self.num == len(self.word_dict) == len(self.words_vocab)
        self.vectors = np.ndarray([self.num, self.word_dim], dtype='float32') # 创建一个shape为(self.num,self.word_dim)的numpy矩阵
        
    def __read_pretrained_word_vecs(self):
        num = 0
        word_dict = {}
        word_dict['<unk>'] = self.num  # unknown word
        with open(self.file_path, 'r', encoding='utf-8') as file:
            file = file.readlines()
            vectors = np.ndarray([len(file) + 1, self.word_dim], dtype='float32')
            vectors[0] = np.random.normal(0.0, 0.3, [self.word_dim]) #unk
            for line in file:
                line = line.split()
                num += 1
                word_dict[line[0]] = num
                vectors[num] = np.asarray(line[-self.word_dim:], dtype='float32')


        for word, idx in self.word_dict.items():
            if idx == 0: continue
            if word in word_dict:
                key = word_dict[word]
                self.vectors[idx] = vectors[key]
            else: self.vectors[idx] = vectors[0]
        self.init_vector = vectors[0]
    
    def __len__(self):
        return self.num

    def get_index(self, word: str):
        if self.word_dict.get(word) is None:
            return 0  # unknown word
        return self.word_dict[word]

    def get_word(self, index:int):
        return self.words_vocab[index]

    def get_vec(self, index: int):
        assert self.vectors is not None
        return self.vectors[index]
    
    
#  对单句数据进行处理的类   
class Sentence():
   
    def __init__(self, dataset_name, tokenizer, vocab):
           self.dataset_name = dataset_name
           self.labels_num = config_data[dataset_name].labels_num
           self.maxlen = config_data[dataset_name].padding_maxlen
           self.vocab = vocab
           self.tokenizer = tokenizer if tokenizer else Tokenizer('normal', remove_stop_words=False)     

    """
    1. 语句分词
    2. 处理成保留长度
    3. 编码
    """
    def data2token(self, sentence):
        # self.data的格式是[sentence1,sentence2....]
        logging.info(f'sentence is to tokens!')
        tokens = self.tokenizer(sentence)# data_token: [tok1,tok2....]
        return tokens
        
    def __encode_tokens(self, tokens):
        '''
        if one sentence length is shorter than maxlen, it will use pad word for padding to maxlen
        :param tokens:
        :return:
        '''
        """对长度不够的sen进行填充(pad:0), 保留长度：maxlen"""
 
        pad_word = 0
        x = [pad_word for _ in range(self.maxlen)] # 声明一个maxlen长度的0列表
        temp = tokens[:self.maxlen] # 保留maxlen长度后的语句->temp
        for idx, word in enumerate(temp):
            x[idx] = self.vocab.get_index(word) # 从词库中获取token对应的index,将[tok1,tok2..]转换成[index1,index2...]
        return torch.tensor(x)
    
    def process(self,sentence):
        tokens = self.data2token(sentence)
        seq =  self.__encode_tokens(tokens)
        return seq
    

class augmentDataReader:
    def __init__(self, filename):
        self.filename = filename
        self.augmentDict = {}
        self.augmentKeys = [] # 收集训练集中所有原样本
        
        self.read_csv() # 读取训练数据集
        
        for key, value_set in self.augmentDict.items():
                self.augmentKeys.append((key))
                
    def read_csv(self):
        with open(self.filename, 'r', newline='') as file:
            reader = csv.reader(file)
            ##################将增强样本set识别成一个字符串了########
            try:
                for row in reader:
                    key = row[0]  # 假设第一列是键
                    value = set(ast.literal_eval(row[1])) # 需要将字符串转成set
                    self.augmentDict[key] = value
            except (ValueError, SyntaxError) as e:
                        print(f'解析错误: {e}')
                        print('row:', key)
                        print('value', value)
                        print('row[1]', row[1])
            except KeyError as e:
                print(f'KeyError: {e}')
                print('row:', key)
                print('value', value)
                print('row[1]', row[1])
            except Exception as e:
                print(f'其他错误: {e}')
                print('row:', key)
                print('value', value)
                print('row[1]', row[1])
    def getAugmentSample(self, statement):
        """根据传入的样本从增强样本csv文件中获取对应的增强样本列表"""
        augmentSample = []
        try:
            if isinstance(statement, str):
                return random.choice(list(self.augmentDict[statement]))
        except:
            print('*******報錯*****')
            print(statement)
            print(len(list(self.augmentDict[statement])))
            print('**************')
    
            for sen in statement:   
                if sen in self.augmentKeys:
                    # 如果找到匹配的键，则从值集合中获取一个元素
                    augmentSample.append(random.choice(list(self.augmentDict[sen])))
                else:
                    raise Exception("no matching augmented data...")

        return augmentSample
                


   