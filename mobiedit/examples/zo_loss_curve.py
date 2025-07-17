import re
import matplotlib.pyplot as plt
import os
import glob
import json

log_dir = 'hparams_log'  # log文件目录
log_files = glob.glob(os.path.join(log_dir, '*.log'))
group_count_lt100 = 0
group_count_100_200 = 0
group_count_200_300 = 0
group_count_300_400 = 0
group_count_400_500 = 0
group_count_500_600 = 0

for file_path in log_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = './loss_curve'
    output_path = f'{output_dir}/{base_name}_loss_curve.pdf'
    group_steps = []
    
    # 检查目录和文件是否存在
    if os.path.exists(output_path):
        # print(f"图表已存在: {output_path}")
        continue
        
    # 创建输出目录(如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取文件名(不含扩展名)用于图片命名
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # 1. 从文件中读取数据
    # file_path = 'test_fg_lr5e-2.log'  # 替换为你的文件路径

    iter_nums = []  # 存储迭代次数
    zo_losses = []  # 存储ZO loss
    edit_succs = []
    group_edit_succs = [] 
    current_group_edit_succ = None

    # 定义正则表达式模式
    pre_pattern = r"'pre':\s*{'rewrite_acc':\s*\[(\d+\.\d+)\]"
    post_pattern = r"'post':\s*{'rewrite_acc':\s*\[(\d+\.\d+)\]"
    
    plt.figure()

    # 2. 打开文件并读取每行数据
    with open(file_path, 'r') as file:
        print(f"Reading file: {file_path}")
        pre_accs = []
        post_accs = []
        rewrite_count = 0
        group_count = 0
        for line in file:
            # 匹配包含rewrite_acc的行
            if "'rewrite_acc': [" in line:
                rewrite_count += 1
                # 每两次匹配只处理第一次
                if rewrite_count % 2 == 1:
                    pre_match = re.search(r"'pre':\s*{'rewrite_acc':\s*\[(\d+\.\d+)\]", line)
                    post_match = re.search(r"'post':\s*{'rewrite_acc':\s*\[(\d+\.\d+)\]", line)
                    
                    if pre_match and post_match:
                        pre_accs.append(float(pre_match.group(1)))
                        post_accs.append(float(post_match.group(1)))

                # 在文件第62行左右，修改edit_succ提取部分
            edit_succ_match = re.search(r"compute v Edit succ: {'rewrite_acc': \[(\d+\.\d+)\]", line)
            if edit_succ_match:
                # print(f"Edit succ: {line}")
                # print(f"edit_succ_match: {edit_succ_match}")
                edit_succ_value = float(edit_succ_match.group(1))
                edit_succs.append(edit_succ_value)
                # current_group_edit_succ = edit_succ_value  # 更新当前组的edit_succ值


            # 3. 使用正则表达式提取Iter和ZO loss
            match = re.match(r"Iter (\d+)/\d+, ZO loss=([0-9.]+)", line)
            # match = re.match(r"\[(\d+)/\d+\]\s*Loss=([0-9.]+)", line)
            # match = re.search(r"loss\s+([0-9.]+)", line)
            if match:
                iter_num = int(match.group(1))
                zo_loss = float(match.group(2))

                if iter_num == 0 and iter_nums:
                    group_steps.append(len(iter_nums)) 
                    # 绘制当前曲线和对应的acc点
                    plt.plot(iter_nums, zo_losses, label=f'Group {group_count+1}', 
                           marker='o', linewidth=1, markersize=1)
                    # 为每个点添加edit_succ值标注（新增）
                    # if group_count < len(group_edit_succs):
                    # print(f"edir_succs.length: {len(edit_succs)}")
                    for i, (x, y) in enumerate(zip(iter_nums, zo_losses)):
                        # print(f"i: {i}, x: {x}, y: {y}")
                        if len(edit_succs) > i:
                            # print(f"edit_succs: {edit_succs[i]}")
                        # print(f"i: {i}, x: {x}, y: {y}")
                        # 每隔几个点添加一次标注，避免过于拥挤
                            if i % 20 == 0 or i == len(iter_nums) - 1:  # 每20个点或最后一个点
                                plt.annotate(f'{edit_succs[i]:.1f}', 
                                        xy=(x, y), 
                                        xytext=(x, y * 1.05),
                                        ha='center',
                                        fontsize=8)

                    
                    # 添加当前组的pre/post acc点
                    if group_count < len(pre_accs):
                        plt.scatter([0], [zo_losses[0]], color='red', marker='*', 
                                  s=100, label=f'Pre Acc {group_count+1}: {pre_accs[group_count]}')
                        plt.scatter([iter_nums[-1]], [zo_losses[-1]], color='green', 
                                  marker='*', s=100, label=f'Post Acc {group_count+1}: {post_accs[group_count]}')
                    
                    group_count += 1
                    plt.xlabel('Iteration')
                    plt.ylabel('ZO loss')
                    plt.title('Loss Curve')
                    plt.grid(True)
                    iter_nums = []  # 清空之前的迭代次数
                    zo_losses = []  # 清空之前的损失值
                    edit_succs = []  # 清空之前的edit_succ值

                # 收集新的数据
                iter_nums.append(iter_num)
                zo_losses.append(zo_loss)

        # 5. 绘制最后一组曲线（防止文件以最后的Iter 0/599结束而没有绘制）
        if iter_nums:
            group_steps.append(len(iter_nums))  
            # 绘制当前曲线和对应的acc点
            plt.plot(iter_nums, zo_losses, label=f'Group {group_count+1}', 
                   marker='o', linewidth=1, markersize=1)
            print(f"edir_succs.length: {len(edit_succs)}")
            
            for i, (x, y) in enumerate(zip(iter_nums, zo_losses)):
                        # print(f"i: {i}, x: {x}, y: {y}")
                        # print(f"edit_succs: {edit_succs[i]}")
                        if len(edit_succs) > i:
                            # print(f"edit_succs: {edit_succs[i]}")
                        # print(f"i: {i}, x: {x}, y: {y}")
                        # 每隔几个点添加一次标注，避免过于拥挤
                            if i % 20 == 0 or i == len(iter_nums) - 1:  # 每20个点或最后一个点
                                plt.annotate(f'{edit_succs[i]:.1f}', 
                                            xy=(x, y), 
                                            xytext=(x, y * 1.05),
                                            ha='center',
                                            fontsize=8)
            
            # 添加当前组的pre/post acc点
            if group_count < len(pre_accs):
                plt.scatter([0], [zo_losses[0]], color='red', marker='*', 
                          s=100, label=f'Pre Acc {group_count+1}: {pre_accs[group_count]}')
                plt.scatter([iter_nums[-1]], [zo_losses[-1]], color='green', 
                          marker='*', s=100, label=f'Post Acc {group_count+1}: {post_accs[group_count]}')
                    
            # plt.plot(iter_nums, zo_losses, label=f'Iteration Group {len(iter_nums)}', marker='o', linewidth=1, markersize=1)

    # 1. 调整图形大小
    plt.gcf().set_size_inches(12, 6)  # 增加图形宽度
    
    # 2. 调整布局，为图例留出空间
    plt.subplots_adjust(right=0.85)  # 给右侧留出15%的空间
    
    # 3. 设置图例位置
    plt.legend(bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.)
    
    # 4. 紧凑布局
    plt.tight_layout()

    # 保存图像到指定文件路径
    plt.savefig(f'./loss_curve/{base_name}_loss_curve.pdf')

    plt.show()
    plt.close()  # 关闭当前图表
    print(f"group_steps: {group_steps}")
    print(f"group_steps: {group_steps}")
    # 统计分档
    for step in group_steps:
        if step < 100:
            group_count_lt100 += 1
        elif 100 <= step < 200:
            group_count_100_200 += 1
        elif 300 <= step < 400:
            group_count_300_400 += 1
        elif 400 <= step < 500:
            group_count_400_500 += 1
        elif 500 <= step <= 600:
            group_count_500_600 += 1
print('统计全部 group_steps 分档：')
print(f'<100      : {group_count_lt100}')
print(f'100-199   : {group_count_100_200}')
print(f'200-299   : {group_count_200_300}')
print(f'300-399   : {group_count_300_400}')
print(f'400-499   : {group_count_400_500}')
print(f'500-600   : {group_count_500_600}')
