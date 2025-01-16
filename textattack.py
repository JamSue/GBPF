"""  
尝试用textattack攻击包攻击训练好的模型
（1）测试GBTD需要用粒球进行投票，怎么使用攻击包加载
"""
import textattack
from textattack.attack_recipes import PWWSRen2019,TextFoolerJin2019,BAEGarg2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
from textattack.attack_results import SuccessfulAttackResult
from textattack.shared.attacked_text import AttackedText
from textattack import Attacker

import os
import logging
import csv
import random
import torch
from config import args,BertConfig,config_data
from LM import myModel
from dataForBert import Sentence
import numpy as np
"""
命令：python textattack.py --pretrained_model_path --model --dataset --attack_method --is_DP
"""

# 设置随机种子，以确保实验的可复现性
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

if args.device== torch.device('cuda'):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' 
    
# （1）创建log文件夹
if not os.path.exists('logs'):
    os.makedirs('logs')
# （2）设置日志记录器
# 清理已有 handlers
root_logger = logging.getLogger()
for h in root_logger.handlers:
    root_logger.removeHandler(h)
    
log_file = f'./logs/textattack_{args.model}_{args.dataset}_{args.attack_method}_info.log' # 日志文件名称
with open(log_file, 'a', encoding="utf-8") as file:
    pass  # 空操作
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("start attack")
###抽样并改成textattack需要的格式
file_path = config_data[args.dataset].test_data_path # 测试数据路径
sample_num = 1000  #抽样数量,具体攻击数量改了textattack包下的某个参数从10->1000

# logging.info("start attack! target_model_path is {} !!!".format(args.pretrained_model_path))

def load_dataset_from_csv(file_path, num_samples=1000):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过 CSV 文件的标题行
        lines = list(reader)
        indices = list(range(len(lines)))
        random_indices = random.sample(indices, min(num_samples, len(lines)))  # 随机抽样不重复的索引
        for index in random_indices:
            row = lines[index]
            label = int(row[0])  # 第一列是标签
            # 将多列合并成单个文本列，保留引号和逗号
            text = '"' + '","'.join(row[1:]) + '"'
            dataset.append((text,label))
    return dataset

dataset = textattack.datasets.Dataset(load_dataset_from_csv(file_path,sample_num))

# 创建一个文本分类任务的模型包装器
class myModelWrapper(ModelWrapper):
    def __init__(self):
        # 初始化模型
        hidden_size = BertConfig.num_hiddens[args.dataset]
        word_dim = BertConfig.word_dim[args.dataset]
        label_num = config_data[args.dataset].labels_num 
        self.model = myModel(hidden_size, word_dim,label_num, dropout=0.1)
        ##加载防御模型和目标模型的权重
        if(torch.cuda.is_available()):
                    Model_weight = torch.load(args.pretrained_model_path) # 如果显卡可用 将模型加载到其训练时的存储设备中 gpu或者cpu
        else:
                Model_weight = torch.load(args.pretrained_model_path,'cpu') # 显卡不可用则加载到cpu上

    ### 多卡训练的模型需开启这段代码
        if args.is_DP:
            state_dict = Model_weight['model_state_dict']

            # 检查key是否有module前缀
            has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())

            if has_module_prefix:
                # 移除'module'前缀
                new_state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
            else:
                new_state_dict = state_dict

            # print("####", new_state_dict)

            # 加载处理后的权重
            self.model.load_state_dict(new_state_dict, strict=False)
        # self.model.load_state_dict(Model_weight['model_state_dict'])
        self.model.to(args.device)
        self.model.eval()

    def __call__(self, text_input_list):
        # 将文本输入传递给您的模型并返回输出,需要处理一句话和多句话的情况
        sen_num = len(text_input_list)
        logit = None
        for i in range(sen_num):
            sen = text_input_list[i]
            processor = Sentence(args.dataset)
            output = processor.process(sen)
            with torch.no_grad():
                if args.model == 'RoBERTa':
                    outputs = self.model(output["input_ids"],output["attention_mask"],
                                        torch.empty([1,300]),target_label=None,flag=1,purity=1)
                else:    
                    outputs = self.model(output["input_ids"],output["attention_mask"],
                                        output["token_type_ids"],target_label=None,flag=1,purity=1)
                
                if logit==None:
                    logit = outputs
                else:
                    logit = torch.cat((logit, outputs), dim=0)
        return logit  # 返回模型输出


# 创建模型包装器
model_wrapper = myModelWrapper()

# 创建 三种 攻击对象
if args.attack_method=='TextFooler':
    textfooler_attack = TextFoolerJin2019.build(model_wrapper)
    attacker_textfooler = Attacker(textfooler_attack, dataset)
    results_textfooler = attacker_textfooler.attack_dataset()
    success_count_textfooler = sum(isinstance(result, SuccessfulAttackResult) for result in results_textfooler)
    logging.info("TextFooler 攻击成功次数：",success_count_textfooler)
    logging.info(results_textfooler)
    
elif args.attack_method=='BertAttack':
    bert_attack = BAEGarg2019.build(model_wrapper)
    logging.info("bert_attack is running .....")
    attacker_bert = Attacker(bert_attack, dataset)
    results_bert = attacker_bert.attack_dataset()
    success_count_bert = sum(isinstance(result, SuccessfulAttackResult) for result in results_bert)

    logging.info("BERTAttack 攻击成功次数：",success_count_bert)
    logging.info(results_bert)
    
elif args.attack_method=="PWWS":
    pwws_attack = PWWSRen2019.build(model_wrapper)
    logging.info("pwws is running .....")
    attacker_pwws = Attacker(pwws_attack, dataset)
    results_pwws = attacker_pwws.attack_dataset()
    success_count_pwws = sum(isinstance(result, SuccessfulAttackResult) for result in results_pwws)
    logging.info("PWWS 攻击成功次数：",success_count_pwws)
    logging.info(results_pwws)
# pwws_attack = PWWSRen2019.build(model_wrapper)


# # # 创建攻击者并运行攻击
# # 


# logging.info("pwws is running .....")
# attacker_pwws = Attacker(pwws_attack, dataset)
# results_pwws = attacker_pwws.attack_dataset()
# success_count_pwws = sum(isinstance(result, SuccessfulAttackResult) for result in results_pwws)
# logging.info(f"PWWS 攻击成功次数：{success_count_pwws}")
# print(results_pwws)


print("Attack finished!",args.model,args.dataset,args.attack_method)