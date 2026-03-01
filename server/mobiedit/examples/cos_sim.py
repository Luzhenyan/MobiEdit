# import re
# import csv
# import os
# import glob

# # 定义日志文件路径
# log_file = '/home/pcllzy/EasyEdit/examples/hparams_log/test_v5_eval_lr5e-2_5.log'
# output_csv = 'cos_sim_results.csv'

# # 存储结果的列表
# results = []

# # 正则表达式模式
# iter_pattern = re.compile(r'Iter (\d+)/(\d+)')
# cos_sim_pattern = re.compile(r'cos similarity of real grad and est grad:([\d.-]+)')

# # 读取日志文件
# with open(log_file, 'r') as f:
#     lines = f.readlines()
    
#     current_iter = None
    
#     for line in lines:
#         # 寻找迭代号
#         iter_match = iter_pattern.search(line)
#         if iter_match:
#             current_iter = int(iter_match.group(1))
        
#         # 寻找余弦相似度
#         cos_sim_match = cos_sim_pattern.search(line)
#         if cos_sim_match and current_iter is not None:
#             cos_sim = float(cos_sim_match.group(1))
#             results.append({'iteration': current_iter, 'cosine_similarity': cos_sim})

# # 按迭代号排序
# results.sort(key=lambda x: x['iteration'])

# # 写入CSV文件
# with open(output_csv, 'w', newline='') as csvfile:
#     fieldnames = ['iteration', 'cosine_similarity']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
#     writer.writeheader()
#     for result in results:
#         writer.writerow(result)

# print(f"已提取{len(results)}条数据并保存到{output_csv}")

# import pandas as pd
# import matplotlib.pyplot as plt
# # import seaborn as sns

# # 读取CSV文件
# df = pd.read_csv('/home/pcllzy/EasyEdit/examples/cos_sim_results.csv')

# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# # 设置图表风格
# # sns.set_style("whitegrid")
# plt.figure(figsize=(12, 6))

# # 为每个迭代号计算平均余弦相似度
# avg_cos_sim = df.groupby('iteration')['cosine_similarity'].mean().reset_index()

# # 绘制主曲线 - 平均值
# plt.plot(avg_cos_sim['iteration'], avg_cos_sim['cosine_similarity'], 
#          color='blue', linewidth=2, label='平均余弦相似度')

# # 添加原始数据点的散点图，透明度低
# plt.scatter(df['iteration'], df['cosine_similarity'], 
#             color='grey', alpha=0.1, s=10, label='单次测量值')

# # 添加标题和标签
# plt.title('余弦相似度随迭代次数的变化', fontsize=16)
# plt.xlabel('迭代次数 (Iteration)', fontsize=14)
# plt.ylabel('余弦相似度 (Cosine Similarity)', fontsize=14)
# plt.savefig('/home/pcllzy/EasyEdit/examples/cos_sim_results.pdf')

import numpy as np
import pandas as pd

# 读取CSV文件中的数值
filepath = "/home/pcllzy/EasyEdit/examples/final_cosine_similarity_fg3_0407.csv"

try:
    # 跳过第一行注释，读取CSV文件
    data = np.loadtxt(filepath, skiprows=1)
    
    # 使用nansum和nanmean函数忽略nan值进行计算
    total_sum = np.nansum(data)
    mean_value = np.nanmean(data)
    
    # 计算非nan值的数量
    valid_count = np.sum(~np.isnan(data))
    total_count = len(data)
    
    # print(f"总和: {total_sum:.6f}")
    print(f"均值: {mean_value:.6f}")
    # print(f"有效数据点: {valid_count}/{total_count} (有 {total_count-valid_count} 个nan值)")
    # print(f"最大值: {np.nanmax(data):.6f}")
    # print(f"最小值: {np.nanmin(data):.6f}")
    # print(f"标准差: {np.nanstd(data):.6f}")
    
    # 可选: 显示nan值的位置
    nan_indices = np.where(np.isnan(data))[0]
    # print(f"nan值的索引位置: {nan_indices}")
    
except Exception as e:
    print(f"发生错误: {e}")