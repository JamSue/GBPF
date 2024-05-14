import random
import torch
import numpy as np
from config import args
import gb_accelerate_temp as new_GBNR

random.seed(42)

"""
调用粒球聚类,返回聚类后的中心centers
input_: out [batch_size,130] [purity,label,hidden_dim]
"""
 
class GBNR(torch.autograd.Function):
   @staticmethod
   def forward(self, input_):
       ### input_: (batch_size,hidden_dim+2)  =>purity+label+feature
         
         self.batch_size = input_.size(0)
         self.embedding_dim = input_.size(1) # purity+label+feature的维度
         input_main = input_[:,1:] # [label+features] 
         self.input = input_[:,2:] # [features] backward中使用
         pur = input_[:,0].numpy().tolist()[0] # 从第0维取出纯度

         numbers,balls,center,radius = new_GBNR.main(input_main,pur) # 数据进行了一次粒球聚类
        
         """
        numbers:    每个球内部样本数
        balls:      [ball_samples*array([球内样本数*[ball_label+64维样本向量]])]
        center:     [ball_numbers*list[ball_label+64维球中心向量]]
        radius:     [ball_numbers,每个球半径]
         """

        
        # 用于存放合格球（球内样本数、样本特征、球的中心向量）
         numbers_qualified = []
         balls_qualified = []
         centers_qualified = []

        # 用于存放需再次聚类的单样本
         sample_for_recluster = []
         index = 0               # 计数
         for ball in balls:      # 取出每个球里面的样本  array([球内样本数*[ball_label+样本特征64维]])
           if numbers[index] < 2:# 将所有单样本保存到sample_for_recluster
                for sample in ball:
                    sample_for_recluster.append(sample)
           else: # 将合格球的参数分别添加(球内样本数、样本特征、球的中心向量）)
                balls_qualified.append(ball)
                numbers_qualified.append(numbers[index])
                centers_qualified.append(center[index])
                index += 1

         # 对单样本重新聚类 降低纯度  有可能存在不需要重新聚球的情况
         if(len(sample_for_recluster)):
            numbers, balls, center, _ = new_GBNR.main(torch.Tensor(np.array(sample_for_recluster)),args.re_pur) 

             # 重新聚类的球都加入合格的球里面
            index = 0
            for ball in balls:
                balls_qualified.append(ball)
                numbers_qualified.append(numbers[index])
                centers_qualified.append(center[index])
                index+=1


         ######## 依据各球内样本数对number、center、balls进行重新排序(冒泡排序): 从大到小
        
        # t = time.time()
         for i in range(1, len(numbers_qualified)):
            for j in range(len(numbers_qualified)-i):
                if numbers_qualified[j] < numbers_qualified[j+1]:
                    numbers_qualified[j], numbers_qualified[j+1] = numbers_qualified[j+1], numbers_qualified[j]
                    balls_qualified[j], balls_qualified[j+1] = balls_qualified[j+1], balls_qualified[j]
                    centers_qualified[j], centers_qualified[j+1] = centers_qualified[j+1], centers_qualified[j]

        #### 将聚好的球中心与标签统计 然后返回
         center_data = []
         center_label = []
         for i in range(len(centers_qualified)):
            #   if(i>0.5*len(centers_qualified)):
            #       break
              center_data.append(centers_qualified[i][1:])
              center_label.append(centers_qualified[i][0])
               
         ## 供backward方法调用
         self.balls = balls_qualified # 各球内样本
         self.numbers = numbers_qualified # 各球内样本数
         self.center = centers_qualified # 各球中心点

     #     print(torch.Tensor(center_data).size()) # ([25,128]) ([ball_num, hidden_dim])
     #     print(torch.Tensor(center_label).size()) # ([25])    (ball_num)
     
         return  torch.Tensor(center_data).to(args.device),torch.Tensor(center_label).to(args.device)

   @staticmethod
   ##  output_grad:从分类器回传到粒球的梯度 shape(balls_number,word_dim) 只有word_dim部分需要反向回传
   def backward(self,output_grad,input): 
         balls = np.array(self.balls,dtype=object)  # shape:[ball_sample_numbers,ball_label+features]

         result = np.zeros([self.batch_size, self.embedding_dim],dtype='float64') # 梯度初始化  形状和输入粒球的数据形状保持一致
         for i in range(output_grad.size(0)):
               for a in balls[i]: # 遍历球的每一条样本
                    # (output_grad[i, :]).to(args.cpu_device)
                    # print("self.input.device",self.input.device)
                    # print("output_grad: ",output_grad[i, :].to(args.cpu_device).device)
                    # print("a:",a.device)
                    result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 2:] = np.array(output_grad[i, :].to(args.cpu_device))
                                                          # 用每个球心向量的梯度去更新球内所有样本的的梯度

         return torch.Tensor(result)
    

"""
note：
怎么用球心向量梯度去更新球内样本的梯度: 
 result[np.where((np.array(self.input) == a[1:]).all(axis=1)), 1:] = np.array(output_grad[i, :])
 (1)self.input() => shape:(batchsize,word_dim)   a[1:]=> 球内的一条样本的特征 [word__dim]
 (2)np.array(self.input) == a[1:] 返回的是与input相同形状的布尔型数组 (batchsize,word_dim)
 (3)(np.array(self.input) == a[1:]).all(axis=1) =>将布尔型数组按照行进行逻辑与操作 即每一行所有的特征位置都为true时才返回true
 (4)np.where((np.array(self.input) == a[1:]).all(axis=1)) => 将为true的下标返回 即将input中处于正在遍历的球的该数据返回true
 (5)result[index,2:] = np.array(output_grad[i, :]) => 将result中word_dim部分的梯度用该球的球心梯度代替

"""