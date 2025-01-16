'样本与正/负粒球中心，不抽取正负样本 进行度量学习'
import os
import logging
import numpy as np
import torch
import random
import torch.nn as nn
from config import args, config_data, BertConfig, LSTMConfig, TextCNNConfig
from visdom import Visdom
from torch.utils.data import DataLoader
from tool import strs2seq, Vote_Neark_label, \
    save_checkpoint  # save_checkpoint,save_checkpoint_withoutGB, Vote_Neark_label,get_pre_balls
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from dataForBert import MyDataset
from LM import myModel, GBQueue
import torch.nn.functional as F
# from augment import textattack_augment
from loss import TripleLoss
from transformers import BertTokenizer, AutoTokenizer, XLNetTokenizer
from dataProcess import augmentDataReader
import torch.nn as nn

# 设置随机种子，以确保实验的可复现性
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别


seed_torch()

device = args.device
if device == torch.device('cuda'):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 控制内存分配策略，避免内存碎片化
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0")
if args.model == 'Bert':
    BERT_PATH = './bert'
    Tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

elif args.model == 'RoBERTa':
    MODEL_PATH = './roberta'
    Tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

elif args.model == 'XLNet':
    MODEL_PATH = './xlnet'
    Tokenizer = XLNetTokenizer.from_pretrained(MODEL_PATH)

### 创建文件储存训练信息

# （1）创建log文件夹
if not os.path.exists('logs'):
    os.makedirs('logs')

# 清理已有 handlers
root_logger = logging.getLogger()
for h in root_logger.handlers:
    root_logger.removeHandler(h)

log_file = f'./logs/trainingBertDP_ab_dataAugment_{args.model}_{args.dataset}_info.log'  # 日志文件名称
with open(log_file, 'a', encoding="utf-8") as file:
    pass  # 空操作
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if args.is_visdom:
    env_name = args.dataset + '_' + args.model  # 可视化界面的窗口名
    vis = Visdom()
    assert vis.check_connection()
    opts = dict(
        title='loss',
        xlabel='epoch',
        legend=['train_loss']
    )

maxlen = config_data[args.dataset].padding_maxlen  # 数据的保留长度
label_num = config_data[args.dataset].labels_num  # 数据的标签数

# 读取数据集
train_data = MyDataset(args.dataset, is_train=True, data_path=config_data[args.dataset].train_data_path)
test_data = MyDataset(args.dataset, is_train=False, data_path=config_data[args.dataset].test_data_path)

test_data.token2seq(maxlen)
train_iterator = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_iterator = DataLoader(test_data, batch_size=args.batch_size * 2, shuffle=True)  # 测试数据量比训练数据加倍
logging.info('{} model initializing...'.format(args.model))

hidden_size = BertConfig.num_hiddens[args.dataset]
word_dim = BertConfig.word_dim[args.dataset]
model = myModel(hidden_size, word_dim, label_num, dropout=0.1)

model.to(device)
model = nn.DataParallel(model,[0,1])

### 初始化模型参数
# 从 DataParallel 对象中获取包装的模型
wrapped_model = model.module
optim_configs = [{'params': wrapped_model.myModel.parameters(), 'lr': args.lr * 0.01},
                 {'params': wrapped_model.encoder.parameters(), 'lr': args.lr},
                 {'params': wrapped_model.classifier.parameters(), 'lr': args.lr}]

optimizer = Adam(optim_configs, lr=1e-4)
# 初始化学习率调节器
lr_scheduler = MultiStepLR(optimizer, milestones=[6, 10, 15], gamma=0.5)

Loss = TripleLoss(label_num)

### 初始化指标和训练中的参数
best_accuracy = 0.0
current_step = 0
ballData_path = './gb_data/{}_{}_ballData.npy'.format(args.dataset, args.model)
best_model_path = "./models/best_model.pth"

logging.info("start training!!!")

gbQueue = GBQueue(args.ball_threshold)  # 粒球队列，最大长度根据数据集调整


# def getPostiveSample(texts, labels):
#     """实现输入的批数据，每条样本随机选取同标签样本作为正样本，如果没有则选自己作为正样本"""
#     label_dict = {}  # 按照标签分样本的字典
#     positiveSample = []  # 对每条样本获取的正样本列表

#     # 按照标签将样本分开
#     for text, label in zip(texts, labels):
#         label = label.item()
#         if label not in label_dict:
#             label_dict[label] = []
#         label_dict[label].append(text)

#     # 遍历每个标签的样本列表，获取正样本
#     for text, label in zip(texts, labels):
#         label = label.item()
        
#         # 检查当前标签的正样本列表
#         same_label_samples = label_dict.get(label, [])
        
#         if len(same_label_samples) > 1:
#             # 如果有多个相同标签的样本，则随机选择一个作为正样本
#             positiveSample.append(random.choice(same_label_samples))
#         else:
#             # 如果没有其他同标签的样本，则返回当前样本本身作为正样本
#             positiveSample.append(text)
            
#     return positiveSample


# def getNegtiveSample(texts, labels):
#     label_dict = {}  # 按照标签分样本的字典
#     negtiveSample = []  # 对每条样本获取的负样本列表
#     # 按照标签将样本分开
#     for text, label in zip(texts, labels):
#         label = label.item()
#         if label not in label_dict:
#             label_dict[label] = []
#             label_dict[label].append(text)
#         else:
#             label_dict[label].append(text)
#     # 遍历每个标签的样本列表，随机选择一条与当前样本标签不同的样本
#     for text, label in zip(texts, labels):
#         # 在当前标签以外的标签中随机选择一个
#         negLabel = random.choice([l for l in label_dict.keys() if l != label])

#         if negLabel:
#             # 在选择的不同标签对应的样本列表中随机选择一条样本
#             negtiveSample.append(random.choice(label_dict[negLabel]))
#         else:
#             ##如果没有异类标签，那么添加一个填充字符串,计算距离会得到0
#             negtiveSample.append("[PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]")
#     return negtiveSample


for epoch in range(1, 1 + args.epoch):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0  # 损失
    eval_num = 0  # 评估次数
    logging.info("Epoch {}/{}".format(epoch, args.epoch))

    if epoch == 3:
        best_accuracy = 0.0  # 重新更新best_accuracy

    if epoch < 3:
        # 正常训练
        step = 0
        for batch_index, (texts, label) in enumerate(train_iterator):
       #     print("1label",label.shape)

            current_step += 1
            step += 1
            # positiveSample = getPostiveSample(texts, label)
            # negtiveSample = getNegtiveSample(texts, label)

            features = Tokenizer(texts, max_length=maxlen, padding='max_length', truncation=True, return_tensors='pt')
            # positiveFeature = Tokenizer(positiveSample, max_length=maxlen, padding='max_length', truncation=True,
            #                             return_tensors='pt')
            # negtiveFeature = Tokenizer(negtiveSample, max_length=maxlen, padding='max_length', truncation=True,
            #                            return_tensors='pt')

            # del positiveSample, negtiveSample
            

            label = label.to(device)

            for feature in [features]:
                feature["input_ids"] = (feature["input_ids"].view(args.batch_size, maxlen)).to(
                    device)  # (batch_size,seq_len)
                feature["attention_mask"] = (feature["attention_mask"].view(args.batch_size, maxlen)).to(device)
                if args.model == 'RoBERTa':
                    feature["token_type_ids"] = None
                if args.model == 'Bert' or args.model == 'XLNet':
                    feature["token_type_ids"] = (feature["token_type_ids"].view(args.batch_size, maxlen)).to(
                        device)

            input_x = features['input_ids']
            attention_mask = features['attention_mask']
            token_type_ids = None
            if args.model=='Bert'or args.model=='XLNet':
                token_type_ids=features["token_type_ids"]
            # 前向传播, 调用模型的forward函数 返回 原始数据、球心过分类器的结果、原始标签、聚类后的球心向量和球心标签
            data, output, target_label, ball_center, center_label = model(input_x,attention_mask,token_type_ids, label, flag=0, purity=args.purity)

            

            # 计算正样本距离和负样本距离
            logits = model(input_x,attention_mask,token_type_ids, label, flag=2, purity=args.purity)

            # positiveLogits = model(positiveFeature["input_ids"],positiveFeature["attention_mask"],positiveFeature["token_type_ids"], label, flag=2, purity=args.purity)
            # negtiveLogits = model(negtiveFeature["input_ids"],negtiveFeature["attention_mask"],negtiveFeature["token_type_ids"], label, flag=2, purity=args.purity)
        
            # dis 表示样本与增强样本的平均距离，dis_dif表示样本与其他样本的距离
            # dis = torch.mean(1 - F.cosine_similarity(logits, positiveLogits, dim=1))
            # neg_dis = torch.mean(1 - F.cosine_similarity(logits, negtiveLogits, dim=1))
           
            # 将本batch的数据数据按标签划分为label_num组并求中心
            arr_center = [torch.zeros(ball_center.shape[1], device=device) for _ in range(label_num)]
            num = torch.zeros(label_num, device=device)  # 初始化每个标签的数据数量
           
            for idx in range(data.shape[0]):
                label = target_label[idx].to(torch.int)  # 将原始数据的对应标签转化成int
                arr_center[label] += data[idx]
                num[label] += 1

            for i in range(label_num):
                arr_center[i] /= num[i]

            dis_center_pos = torch.zeros(data.shape[0], device=device)
            dis_center_neg = torch.zeros(data.shape[0], device=device)

          
            for i in range(data.shape[0]):
                # 计算与目标标签中心的距离
                dis_center_pos[i] = 1 - F.cosine_similarity(data[i:i + 1, :], arr_center[target_label[i]], dim=-1)
                # 单数据与其他标签中心的距离 shape: [label]
                dis_center_neg[i] = 1 - torch.mean(F.cosine_similarity(data[i:i + 1, :],
                                                                       torch.stack(
                                                                           [arr_center[j] for j in range(label_num) if
                                                                            j != target_label[i]]), dim=-1))


            dis_center_pos = torch.mean(dis_center_pos, dim=0)
            dis_center_neg = torch.mean(dis_center_neg, dim=0)
    
            cross_entropy_loss = nn.CrossEntropyLoss()
            entropyLoss = cross_entropy_loss(output, center_label.long())
            # loss1 = args.a * (dis + dis_center_pos) - args.b * (neg_dis + dis_center_neg)
            loss1 = args.a *  dis_center_pos - args.b *  dis_center_neg

            loss = loss1 + entropyLoss
            optimizer.zero_grad()  # 梯度清零
            for name, param in model.named_parameters():
                print('here...............',param.grad)
                if param.grad is None:
                    print('存在none的梯度')
                if param.grad is not None:
                    print(f"{name}: {param.grad.mean()}, {param.grad.max()}")

            print('测试点12123') # 梯度消失了 为什么！！？
            loss.backward()  # 反向传播 
            print('Finished backward pass, updating optimizer...')
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()
            if step % 20 == 0:  # 每20次做一次评估
                model.eval()  # 进入测试，测试时不启用 Batch Normalization 和 Dropout
                num_correct = 0  # 记录预测正确的数量
                test_total = 0  # 记录测试的总数
                eval_num += 1
                logging.info("###### Now is a evaluation phase, <evaluation num>:{}  #######".format(eval_num))

                with torch.no_grad():
                    for i, (texts, label) in enumerate(test_iterator):
                        label = label.to(device)  # size: ([64])
                        # features = features.to(device) # texts(bs*2)
                        features = Tokenizer(texts, max_length=maxlen, padding='max_length', truncation=True,
                                             return_tensors='pt')
                        if args.model=='RoBERTa':
                            features["token_type_ids"] = None
                        _, output = model(features['input_ids'],features['attention_mask'],features["token_type_ids"], label, flag=-1, purity=1)

                        _, pred = torch.max(output.data, 1)
                        test_correct = (pred == label).sum().item()
                        num_correct += test_correct
                        test_total += label.size(0)

                    test_accuracy = num_correct / test_total * 100.

                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_acc_loc = epoch
                        best_eval_num_loc = eval_num

                    if current_step != 0: running_loss = running_loss / 20
                    logging.info(
                        '%d-th epoch, total_batch_num:%5d. train_loss: %.4f test_accuracy: %.2f test_correct:%d test_total:%d'
                        % (epoch, i + 1, running_loss, test_accuracy, num_correct, test_total))

                if args.is_visdom:
                    vis.line(X=[current_step], Y=[running_loss], env=env_name,
                             opts=dict(title='loss', legend=['train_loss']), win='loss', name='train_loss',
                             update='append')
                    vis.line(X=[current_step], Y=[test_accuracy], env=env_name,
                             opts=dict(title='test_acc', legend=['test_acc']), win='test_acc', name='test_acc',
                             update='append')
                running_loss = 0.0

                model.train()  # 返回训练
                logging.info('Current best acc:{:.2f}% at epoch{} --eval_num{}'.format(best_accuracy, best_acc_loc,
                                                                                       best_eval_num_loc))
                logging.info('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))

    if epoch >= 3:
        step = 0
        for batch_index, (texts, label) in enumerate(train_iterator):
            current_step += 1
            step += 1
            # positiveSample = getPostiveSample(texts, label)
            # negtiveSample = getNegtiveSample(texts, label)
            
            features = Tokenizer(texts, max_length=maxlen, padding='max_length', truncation=True, return_tensors='pt')
            # positiveFeature = Tokenizer(positiveSample, max_length=maxlen, padding='max_length', truncation=True,
            #                             return_tensors='pt')
            # negtiveFeature = Tokenizer(negtiveSample, max_length=maxlen, padding='max_length', truncation=True,
            #                            return_tensors='pt')
            

            label = label.to(device)

            for feature in [features]:
                feature["input_ids"] = (feature["input_ids"].view(args.batch_size, maxlen)).to(
                    device)  # (batch_size,seq_len)
                feature["attention_mask"] = (feature["attention_mask"].view(args.batch_size, maxlen)).to(device)
                if args.model == 'RoBERTa':
                    feature["token_type_ids"] = None
                if args.model == 'Bert' or args.model == 'XLNet':
                    feature["token_type_ids"] = (feature["token_type_ids"].view(args.batch_size, maxlen)).to(
                        device)

            input_x = features['input_ids']
            attention_mask = features['attention_mask']
            if args.model=='Bert'or args.model=='XLNet':
                token_type_ids=features["token_type_ids"]
            # 前向传播, 调用模型的forward函数 返回 原始数据、球心过分类器的结果、原始标签、聚类后的球心向量和球心标签
            data, output, target_label, ball_center, center_label = model(input_x,attention_mask,token_type_ids, label, flag=0, purity=args.purity)

            # 计算正样本距离和负样本距离
            logits = model(input_x,attention_mask,token_type_ids, label, flag=2, purity=args.purity)
           
         

            # positiveLogits = model(positiveFeature["input_ids"],positiveFeature["attention_mask"],positiveFeature["token_type_ids"], label, flag=2, purity=args.purity)
            
            # del positiveFeature
            
            

            # negtiveLogits = model(negtiveFeature["input_ids"],negtiveFeature["attention_mask"],negtiveFeature["token_type_ids"], label, flag=2, purity=args.purity)
           
            # del negtiveFeature
     
            # dis 表示样本与增强样本的平均距离，dis_dif表示样本与其他样本的距离
            # dis = torch.mean(1 - F.cosine_similarity(logits, positiveLogits, dim=1)) / args.batch_size
            # neg_dis = torch.mean(1 - F.cosine_similarity(logits, negtiveLogits, dim=1))

            # 将本batch的数据数据按标签划分为label_num组并求中心
            arr_center = [torch.zeros(ball_center.shape[1], device=device) for _ in range(label_num)]
            num = torch.zeros(label_num, device=device)  # 初始化每个标签的数据数量

            for idx in range(data.shape[0]):
                label = target_label[idx].to(torch.int)  # 将原始数据的对应标签转化成int
                arr_center[label] += data[idx]
                num[label] += 1

            for i in range(label_num):
                arr_center[i] /= num[i]

            dis_center_pos = torch.zeros(data.shape[0], device=device)
            dis_center_neg = torch.zeros(data.shape[0], device=device)

         
            for i in range(data.shape[0]):
                # 计算与目标标签中心的距离
                dis_center_pos[i] = 1 - F.cosine_similarity(data[i:i + 1, :], arr_center[target_label[i]], dim=-1)
                # 单数据与其他标签中心的距离 shape: [label]
                dis_center_neg[i] = 1 - torch.mean(F.cosine_similarity(data[i:i + 1, :],
                                                                       torch.stack(
                                                                           [arr_center[j] for j in range(label_num) if
                                                                            j != target_label[i]]), dim=-1))


            dis_center_pos = torch.mean(dis_center_pos, dim=0)
            dis_center_neg = torch.mean(dis_center_neg, dim=0)


            cross_entropy_loss = nn.CrossEntropyLoss()
            entropyLoss = cross_entropy_loss(output, center_label.long())
            loss1 = args.a * dis_center_pos - args.b * dis_center_neg 

            loss = loss1 + entropyLoss

            # 将 (ball_center,center_label)拼接 并将每条粒球数据分别压入队列中
            ball_data = (torch.cat((ball_center.cpu().detach(), (center_label.cpu().detach()).unsqueeze(1)), dim=1)).to(
                device)
            gbQueue.push(ball_data)
            gbQueue.pop_if_full()

            ball_num = center_label.shape[0]

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播 通过模型的反向传播函数进行传播的
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()
            # if args.is_visdom:
            #     vis.line(X=[current_step], Y=[ loss.item()], env=env_name,  opts=dict(title='loss', legend=['train_loss']), win='loss', name='train_loss', update='append')

            if step % 100 == 0:  # 每100次做一次评估
                model.eval()  # 进入测试，测试时不启用 Batch Normalization 和 Dropout
                num_correct = 0  # 记录预测正确的数量
                vote_num_correct = 0  # 投票预测正确
                vote_test_total = 0  # 投票测试总数
                test_total = 0  # 记录测试的总数
                eval_num += 1

                logging.info("###### Now is a evaluation phase, <evaluation num>:{}  #######".format(eval_num))

                with torch.no_grad():
                    for i, (texts, label) in enumerate(test_iterator):
                        # LLP=>评估阶段，不过粒球层 用聚好的粒球最近的中心替换进行评估
                        features = Tokenizer(texts, max_length=maxlen, padding='max_length', truncation=True,
                                             return_tensors='pt')
                        label = label.to(device)
                        bs = label.shape[0]
                        features["input_ids"] = features["input_ids"].view(bs, maxlen)  # (batch_size,seq_len)
                        features["attention_mask"] = features["attention_mask"].view(bs, maxlen)
                        if args.model == 'RoBERTa':
                            features["token_type_ids"] = None
                        if args.model == 'Bert' or args.model == 'XLNet':
                            features["token_type_ids"] = features["token_type_ids"].view(bs, maxlen)
                        _, output= model(  features["input_ids"],features["attention_mask"], features["token_type_ids"], label, flag=-1, purity=1)

                        _, pred1 = torch.max(output.data, 1)
                        test_correct = (pred1 == label).sum().item()
                        num_correct += test_correct
                        test_total += label.size(0)

                        logit = model(features["input_ids"],features["attention_mask"],features["token_type_ids"], label, flag=2, purity=1)  # 直接获取过模型不过分类器的logit (2*bs,hidden_size)
                        # 读取聚好的球心和标签
                        balls = gbQueue.concatTensors()
                        loaded_ball_centers = balls[:, :-1].float()
                        loaded_center_labels = balls[:, -1].float()
                        ball_num = loaded_ball_centers.size(0)  # 粒球数量

                        logging.info('###### voting from %d balls ######' % (ball_num))

                        _, pred = Vote_Neark_label(logit, wrapped_model.classifier, loaded_ball_centers, loaded_center_labels,
                                                   args.k)
                        vote_test_correct = (pred.to(device) == label.to(device)).sum().item()
                        vote_num_correct += vote_test_correct
                        vote_test_total += label.size(0)

                    if current_step != 0: running_loss = running_loss / 100
                    test_accuracy = num_correct / test_total * 100.
                    vote_test_accuracy = vote_num_correct / vote_test_total * 100.  # 所有测试数据的投票结果

                    logging.info(
                        "In this eval_batch, num_correct: %d, test_total:%d,the number of GB is %d,test_accuracy: %.5f" % (
                        num_correct, test_total, ball_num, test_accuracy))

                    logging.info(
                        '%d-th epoch, total_batch_num:%d train_loss: %.4f vote_test_accuracy: %.2f vote_test_correct:%d vote_test_total:%d'
                        % (epoch, i + 1, running_loss, vote_test_accuracy, vote_num_correct, vote_test_total))

                    if vote_test_accuracy > best_accuracy:
                        best_accuracy = vote_test_accuracy
                        best_acc_loc = epoch
                        best_eval_num_loc = eval_num
                        save_checkpoint(best_accuracy, args.model, args.dataset, model, optimizer, best_acc_loc)

                        # 将上一轮的球保存到文件中
                        ball_centers_np = loaded_ball_centers.cpu().numpy()  # 转换ball_centers为NumPy数组
                        center_labels_np = loaded_center_labels.cpu().numpy()  # 转换center_labels为NumPy数组
                        ball_data = np.column_stack(
                            (ball_centers_np, center_labels_np))  # 将ball_centers_np和center_labels_np合并为一个二维数组
                        file_path = './gb_data/{}_{}_ballData.npy'.format(args.dataset, args.model)  # 保存的npy文件路径
                        np.save(file_path, ball_data)  # 保存数据为npy文件

                        logging.info('save balls at {%d}th-epoch %dth-eval, best_acc_save:%.5f'
                                     % (epoch, best_eval_num_loc, best_accuracy))
                    if args.is_visdom:
                        vis.line(X=[current_step], Y=[running_loss], env=env_name,
                                 opts=dict(title='loss', legend=['train_loss']), win='loss', name='train_loss',
                                 update='append')
                        vis.line(X=[current_step], Y=[test_accuracy], env=env_name,
                                 opts=dict(title='test_acc', legend=['test_acc']), win='test_acc', name='test_acc',
                                 update='append')
                        vis.line(X=[current_step], Y=[vote_test_accuracy], env=env_name,
                                 opts=dict(title='vote_acc', legend=['vote_acc']), win='vote_acc', name='vote_acc',
                                 update='append')
                    running_loss = 0.0

                model.train()  # 返回训练

                logging.info('Current best acc:{:.2f}% at epoch{} --eval_num{}'.format(best_accuracy, best_acc_loc,
                                                                                       best_eval_num_loc))
                logging.info('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))

    lr_scheduler.step()

logging.info("training end!!!")
# 关闭日志记录器
logging.shutdown()