import csv

###region
# """
# 脚本为将第二三列数据拼接，并删除第三列，并且将数据用引号引用起来
# """
# 读取 CSV 文件并解析数据
# with open('./dataset/AGNEWS/train.csv', 'r', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = list(reader)

# # 将第三列文本合并到第二列
# for row in rows:
#     row[1] = row[1] + ' ' + row[2]

# # 移除第三列
# for row in rows:
#     del row[2]


# # 写入新的 CSV 文件
# with open('./dataset/AGNEWS/train1.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
#     writer.writerows(rows)

###endregion

"""测试代码"""

# # 抽取120000条数据

# def extract_data(input_file, output_file, num_records):
#     # 打开CSV文件进行读取
#     with open(input_file, 'r', newline='', encoding='utf-8') as csv_input:
#         # 使用csv.reader读取CSV文件
#         reader = csv.reader(csv_input)
        
#         # 创建一个写入CSV文件的对象
#         with open(output_file, 'w', newline='', encoding='utf-8') as csv_output:
#             # 使用csv.writer写入CSV文件
#             writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
            
#             # 写入CSV文件的标题行（如果有的话）
#             header = next(reader)
#             writer.writerow(header)
            
#             # 抽取指定数量的记录
#             for i, row in enumerate(reader):
#                 if i < num_records:
#                     writer.writerow(row)
#                 else:
#                     break

# # 调用函数来执行数据抽取
# extract_data('dataset/yelp/train_all.csv', 'dataset/yelp/train.csv', 120000)


### 按照标签抽取数据

import csv
import random

def extract_data(csv_file, num_samples,label_num):
    # 读取CSV文件
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data_dict = {}
        for row in reader:
            label = row[0]
            data = row[1]
            if label not in data_dict:
                data_dict[label] = []
            data_dict[label].append(data)
    
    # 计算每个标签应该抽取的样本数
    samples_per_label = num_samples // label_num
    
    extracted_data = []
    for label, samples in data_dict.items():
        # 如果某个标签下的样本数不足以达到预期数量，则全选
        if len(samples) <= samples_per_label:
            extracted_data.extend([(label, sample) for sample in samples])
        else:
            # 随机抽取指定数量的样本
            extracted_data.extend([(label, sample) for sample in random.sample(samples, samples_per_label)])
    
    return extracted_data

def write_to_csv(data, output_file):
    # 写入到CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        # writer = csv.writer(file)
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        # writer.writerow(["Label", "Data"])  # 写入标题行
        writer.writerows(data)

if __name__ == "__main__":
    csv_file = "dataset/YAHOO/train_all.csv"  # 替换为你的CSV文件路径
    output_file = "dataset/YAHOO/train.csv"  # 输出文件路径
    # csv_file = "dataset/yelp/train_all.csv"  # 替换为你的CSV文件路径
    # output_file = "dataset/yelp/train.csv"  # 输出文件路径
    num_samples = 120000
    
    extracted_data = extract_data(csv_file, num_samples,10)
    write_to_csv(extracted_data, output_file)
