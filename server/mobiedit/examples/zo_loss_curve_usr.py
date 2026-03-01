import re
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({
    'font.size': 18,          # 默认字体大小
    'axes.titlesize': 18,     # 标题字体大小
    'axes.labelsize': 18,     # 坐标轴标签字体大小
    'xtick.labelsize': 18,    # x轴刻度标签字体大小
    'ytick.labelsize': 18,    # y轴刻度标签字体大小
    'legend.fontsize': 16,    # 图例字体大小
    'figure.titlesize': 16    # 图表标题字体大小
})

log_dir = './hparams_log'
n_range = range(1,2)  # n=2,3,4,5,6
log_files = [f'test_qwen2.5_usrv3_zo-step600_1.log' for n in n_range]

plt.figure(figsize=(12, 6))

markers = ['o', 's', '^', 'D', 'v']  # 不同的标记样式

for n, log_file, marker in zip(n_range, log_files, markers):
    file_path = os.path.join(log_dir, log_file)
    print(f"Processing file: {file_path}")
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 - {file_path}")
        continue

    iter_nums = []
    zo_losses = []
    edit_succs = []
    group_count = 0  # 用于计数遇到的group数量
    current_group_data = []  # 临时存储当前group的数据

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 提取loss
                loss_match = re.match(r"Iter (\d+)/\d+, ZO loss=([0-9.]+)", line)
                if loss_match:
                    iter_num = int(loss_match.group(1))
                    zo_loss = float(loss_match.group(2))
                    
                    if iter_num == 0:
                        group_count += 1
                        # 如果是2号文件(n=3)且遇到第2个group，开始记录数据
                        # if n == 2 and group_count == 2:
                        current_group_data = []  # 重置当前group数据
                        # elif n == 2 and group_count > 2:
                        #     break  # 对于2号文件，只需要第2个group
                        # elif n != 2 and group_count > 1:
                        #     break  # 其他文件只需要第1个group
                    
                    # 记录数据条件
                    # if (n == 2 and group_count == 2) or (n != 2 and group_count == 1):
                    current_group_data.append((iter_num, zo_loss))

                # 提取edit_succ
                # if (n == 2 and group_count == 2) or (n != 2 and group_count == 1):
                edit_succ_match = re.search(
                    r"compute v Edit succ: {'rewrite_acc': \[(\d+\.\d+)\]", 
                    line
                )
                if edit_succ_match:
                    edit_succs.append(float(edit_succ_match.group(1)))

            # 处理收集到的数据
            if current_group_data:
                iter_nums, zo_losses = zip(*current_group_data)
                iter_nums = list(iter_nums)
                zo_losses = list(zo_losses)

        # 绘图
        if iter_nums and edit_succs:
            line, = plt.plot(iter_nums, zo_losses, label=f'Fact {n-1}', marker=marker, markevery=20)
            color = line.get_color()
            
            # 从第20个step开始标注
            start_step = 0
            interval = 100  # 采样间隔
            
            for i, succ in enumerate(edit_succs):
                pos = start_step + i * interval
                if pos < len(iter_nums):
                    step = iter_nums[pos]
                    loss = zo_losses[pos]
                    
                    # 在曲线上打点
                    plt.plot(step, loss, marker=marker, color=color, markersize=8)
                    
                    # 添加标注
                    plt.annotate(f'{succ:.2f}',
                                xy=(step, loss),
                                xytext=(step, loss*1.05),
                                fontsize=10,
                                ha='center',
                                color=color
                                )

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        continue

plt.xlim(0, 602)
plt.xlabel('Step')
plt.ylabel('Loss')
# plt.title('ZO Loss with Edit Success Rate Annotations\n(For n=3, showing 2nd group only)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 确保输出目录存在
os.makedirs(os.path.dirname('./qwen2.5_usrv3_zo_600step.pdf'), exist_ok=True)
plt.savefig('./qwen2.5_usrv3_zo_600step.pdf', bbox_inches='tight', dpi=300)
plt.show()