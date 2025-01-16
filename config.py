"""
    数据增强的参数，例如每条样本需要多少条增强样本
    bert系列的模型处理
"""

import os
import torch
import numpy as np
import argparse

config_dataset_list = ['IMDB','AGNEWS','YAHOO','YELP','yelp'] # yelp是2分类 YELP是5分类
config_model_list = ['LSTM', 'TextCNN', 'BidLSTM','Bert','XLNet','RoBERTa']
parser = argparse.ArgumentParser()

parser.add_argument('--cpu_device', default= torch.device("cpu"), help="sometimes need to put datas to the cpu")
parser.add_argument('--device',default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help="device")
parser.add_argument('--dataset',choices=config_dataset_list,default='AGNEWS')
parser.add_argument('--model',choices=config_model_list,default='LSTM')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--lr',type=float,default=1e-3,help="learning rate")  
parser.add_argument('--load_model',choices=[True, False], default=False, type=bool) # 是否加载预训练模型
parser.add_argument('--is_visdom', choices=[True, False], default=False, type=bool,help="use visdom or not")

parser.add_argument('--rate', default=0.25, type=float, help='训练时的掩码比例')
parser.add_argument('--purity',default=1,type=int,help='the purity setting of Granular-Ball while training')
parser.add_argument('--model_train', choices=[True, False], default=False, type=bool, 
                    help='Is the model in training phase, params need backward or not')

parser.add_argument('--a', default=1, type=float, help='Hyperparameters in the loss, hyperparameters of dis') #损失中的超参数，dis的超参
parser.add_argument('--b', default=0.1, type=float, help='Hyperparameters in the loss, hyperparameters of dis_dif') # 损失中的超参数，dis_dif的超参

parser.add_argument('--maskRate', default=0.25, type=float, help='mask ratio during training')# 训练时的掩码比例
parser.add_argument('--augment_num', default=20, type=int, help='augment num of train data')# 训练时的掩码比例
parser.add_argument('--dropout', default=0.1, type=float, help='')

# 粒球聚类的参数
parser.add_argument('--recluster',default=3,type=int,help='聚球次数')
parser.add_argument('--re_pur', default=1,help='重新聚类的纯度')


# 攻击的参数设置
parser.add_argument('--sample_names',default=1000, type=int, help="数据集样本抽样条数")
parser.add_argument('--clean_samples_path',default=None,type=str,help="抽取的干净样本的路径")
parser.add_argument('--attack_method',default="PWWS", type=str,
                     choices=["TextFooler","PWWS", "RES","GA","NoAttack","onlyPre","BertAttack"], help="攻击方式")
parser.add_argument('--sub_rate_limit', type=float, default=0.35) #  替换的限制概率  要是替换了一定概率的词还没有攻击成功 就停止替换
parser.add_argument('--target_model_path',default=''
                    , help="攻击的预训练目标模型的路径") 
parser.add_argument('--pretrained_model_path',default="./models/(bs128)model_LSTM_AGNEWS_best_87.5.pth",
                    type=str, help="预训练好的模型路径")

parser.add_argument('--k', default=4, type=int,help="k值，训练与应用中都需要用到")
parser.add_argument('--count_t', default=3, type=int,help="训练时每个batch保留的球数")
parser.add_argument('--ball_threshold', default=3000, type=int,help="粒球队列中保存数量的阈值")
parser.add_argument('--ball_path', default='',  help="粒球保存路径")
parser.add_argument('--is_DP', default='false',  help="针对textattack攻击，多卡训练与单卡训练配置不同")

args = parser.parse_args()

class IMDBConfig():
    data_dir = r'./dataset/IMDB'
    train_data_path = r'./dataset/IMDB/train.csv'
    test_data_path = r'./dataset/IMDB/test.csv'
    train_data_npy_path = r'./dataset/IMDB/train.npy'
    test_data_npy_path = r'./dataset/IMDB/test.npy'
    attack_data_dir = r'./data_attacked/imdb'
    augment_data_path = r'./dataset/IMDB/augment.csv'
    labels_num = 2
    vocab_limit_size = 80000 # IMDB建立数据集限制的词库大小
    tokenizer_type = 'normal' # IMDB分词方式的选择
    remove_stop_words = False #  为什么不去除停用词
    padding_maxlen = 300 # 保留长度 

    purity = 0.85
class AGNEWSConfig():
    data_dir = r'./dataset/AGNEWS'
    train_data_path = r'./dataset/AGNEWS/train.csv'
    test_data_path = r'./dataset/AGNEWS/test.csv'
    train_data_npy_path = r'./dataset/AGNEWS/train.npy'
    test_data_npy_path = r'./dataset/AGNEWS/test.npy'
    attack_data_dir = r'./data_attacked/agnews'
    augment_data_path = r'./dataset/AGNEWS/augment.csv'
    labels_num = 4
    vocab_limit_size = 80000 
    tokenizer_type = 'normal' 
    remove_stop_words = False 
    padding_maxlen = 50 

    purity = 0.8

class YELPConfig():
    data_dir = r'./dataset/YELP'
    train_data_path = r'./dataset/YELP/train.csv'
    test_data_path = r'./dataset/YELP/test.csv'
    train_data_npy_path = r'./dataset/YELP/train.npy'
    test_data_npy_path = r'./dataset/YELP/test.npy'
    attack_data_dir = r'./data_attacked/yelp'
    augment_data_path = r'./dataset/YELP/augment.csv'
    labels_num = 5
    vocab_limit_size = 80000 
    tokenizer_type = 'normal' # IMDB分词方式的选择
    remove_stop_words = False #  为什么不去除停用词
    padding_maxlen = 500 # 保留长度 

class yelpPolarityConfig():
    data_dir = r'./dataset/yelp'
    train_data_path = r'./dataset/yelp/train.csv'
    test_data_path = r'./dataset/yelp/test.csv'
    train_data_npy_path = r'./dataset/yelp/train.npy'
    test_data_npy_path = r'./dataset/yelp/test.npy'
    attack_data_dir = r'./data_attacked/yelp'
    augment_data_path = r'./dataset/yelp/augment.csv'
    labels_num = 2
    vocab_limit_size = 80000 
    tokenizer_type = 'normal' # IMDB分词方式的选择
    remove_stop_words = False #  为什么不去除停用词
    padding_maxlen = 300 # 保留长度 
class YAHOOConfig():
    data_dir = r'./dataset/YAHOO'
    train_data_path = r'./dataset/YAHOO/train.csv'
    test_data_path = r'./dataset/YAHOO/test.csv'
    train_data_npy_path = r'./dataset/YAHOO/train.npy'
    test_data_npy_path = r'./dataset/YAHOO/test.npy'
    attack_data_dir = r'./data_attacked/yahoo'
    augment_data_path = r'./dataset/YAHOO/augment.csv'
    labels_num = 10
    vocab_limit_size = 80000 
    tokenizer_type = 'normal' # IMDB分词方式的选择
    remove_stop_words = False #  为什么不去除停用词
    padding_maxlen = 300 # 保留长度 
  
config_data ={'IMDB': IMDBConfig, 'AGNEWS':AGNEWSConfig, 'YELP':YELPConfig, 'YAHOO':YAHOOConfig, 'yelp':yelpPolarityConfig}

class LSTMConfig():
    num_hiddens = { 'IMDB': 128,'AGNEWS': 128, 'YAHOO': 128, 'YELP': 128, 'yelp': 128}

    num_layers = {'IMDB': 2, 'AGNEWS': 2, 'YAHOO': 2, 'YELP': 2 , 'yelp': 2}

    is_using_pretrained = { 'IMDB': True, 'AGNEWS': True, 'YAHOO': True,'YELP': True , 'yelp': True}

    word_dim = { 'IMDB': 300, 'AGNEWS': 300, 'YAHOO': 300, 'YELP': 300 , 'yelp': 300}
    
class TextCNNConfig():
    channel_kernel_size = {
        'IMDB': ([50, 50, 50], [3, 4, 5]), 'AGNEWS': ([50, 50, 50], [3, 4, 5]),
        'YAHOO': ([50, 50, 50], [3, 4, 5]), 'YELP': ([50, 50, 50], [3, 4, 5]), 'yelp': ([50, 50, 50], [3, 4, 5]) }
    
    is_static = { 'IMDB': True, 'AGNEWS': True, 'YAHOO': True, 'YELP': True , 'yelp': True}
    
    using_pretrained = {'IMDB': True, 'AGNEWS': True, 'YAHOO': True,'YELP': True, 'yelp': True }

    word_dim = { 'IMDB': 300, 'AGNEWS': 300, 'YAHOO': 300, 'YELP': 300, 'yelp': 300 }
    
### 暂时三个模型公用这一个参数
class BertConfig():
    num_hiddens = {'IMDB':  768,  'AGNEWS':768, 'YAHOO': 768, 'YELP':  768, 'yelp':  768,   }

    word_dim = {'IMDB': 100, 'AGNEWS': 100, 'YAHOO': 100, 'YELP': 100, 'yelp':100 }
    
class ballConfig:
        if not os.path.exists('gb_data'):
            os.makedirs('gb_data')
        # ball_path = './gb_data/{}_{}_ballData.npy'.format(args.dataset,args.model) # 聚球后数据的npy文件路径
        ball_path = args.ball_path if args.ball_path != '' else './gb_data/{}_{}_ballData.npy'.format(args.dataset, args.model)
        if(os.path.exists(ball_path)):
            loaded_data = np.load(ball_path)
            loaded_ball_centers =  torch.from_numpy(loaded_data[:, :-1]).float()
            loaded_center_labels = torch.from_numpy(loaded_data[:, -1] ).float()
            ball_num = loaded_ball_centers.size(0) # 粒球数量
        else:
             loaded_ball_centers = None
             loaded_center_labels = None
 
 ### 攻击要用到的参数
config_pwws_use_NE = False
config_RSE_mask_low = 2 
config_RSE_mask_rate = 0.25

