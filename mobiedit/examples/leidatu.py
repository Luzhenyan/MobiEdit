import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 原始数据
data = {
    'method': ['ROME', 'MEMIT', 'WISE', 'AhphaEdit', 'MobiEdit'],
    'Time': [67.07, 67.07, 146.77, 67.07, 25.51],
    'Memory': [46.1352, 46.1352, 46.3032, 46.1352, 6.2],
    'Energy': [221.93, 221.93, 485.62, 221.93, 18.72],
    'Edit Success': [92.63, 91.00, 100.00, 94.00, 80.12],
    'Locality': [60.36, 99.00, 98.00, 32.00, 72.65],
    'Portability': [47.24, 30.00, 100.00, 91.00, 51.35]
    
}
data = {
    'method': ['ROME', 'MEMIT', 'AhphaEdit', 'Our Method'],
    'Edit Success': [93.6552, 94, 98, 78.1272],
    'Locality': [47.638, 60, 67, 50.7826],
    'Portability': [54.243, 63, 94, 55.8735],
    'Fluency': [606.4415, 612, 622, 624.8937],  # 要反归一化
    'Time': [65.19631053, 65.19631053, 65.19631053, 30.24732929],
    'Memory': [46.1352, 46.1352, 46.1352, 6.2],
    'Energy': [174.3622366, 174.3622366, 348.7244733, 17.63071204]
}

# 加载数据
df = pd.DataFrame(data)

# 创建映射函数：效率指标映射到 [lower_bound, 100]，反归一化
def reverse_map_to_custom_range(x, min_val, max_val, lower_bound=10, upper_bound=100):
    norm = 1 - (x - min_val) / (max_val - min_val)
    return norm * (upper_bound - lower_bound) + lower_bound

# 应用映射到效率类指标
df['Time_eff']   = reverse_map_to_custom_range(df['Time'],   df['Time'].min(),   df['Time'].max())
df['Memory_eff'] = reverse_map_to_custom_range(df['Memory'], df['Memory'].min(), df['Memory'].max())
df['Energy_eff'] = reverse_map_to_custom_range(df['Energy'], df['Energy'].min(), df['Energy'].max())

# 最终用于绘图的维度（全部在 40~100 范围）
plot_cols = [ 'Memory_eff', 'Time_eff', 'Energy_eff','Edit Success', 'Portability', 'Locality']
labels = ['Memory Efficiency', 'Time Efficiency',  'Energy Efficiency','Edit Success', 'Portability', 'Locality']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 创建雷达图画布
fig, ax = plt.subplots(subplot_kw=dict(polar=True))

# 配色方案
colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
markers = ['o', 's', '*', '^', 'D']

# 画每个方法
for i, row in df.iterrows():
    values = row[plot_cols].tolist()
    values += values[:1]
    label = row['method']
    is_our = 'mobi' in label.lower()

    ax.plot(angles, values, label=label, color=colors[i],
            linewidth=2.5 if is_our else 1.5,
            marker=markers[i], markersize=7)
    # ax.fill(angles, values, color=colors[i], alpha=0.2 if is_our else 0.06)

# 设置角标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=15, fontweight='bold')

# 设置径向刻度
ax.set_ylim(0, 100)
ax.set_yticks([40, 60, 80, 100])
ax.set_yticklabels(['40', '60', '80', '100'], fontsize=10)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

# 标题与图例
# plt.title("综合对比：编辑效果与效率指标（统一评分 40~100）", fontsize=14, weight='bold', y=1.08)
# legend = plt.legend(loc='upper left', bbox_to_anchor=(1.25, 1.05), fontsize=10)
# # 加粗我们方法的图例项
# for text in legend.get_texts():
#     if 'our' in text.get_text().lower():
#         text.set_fontweight('bold')
# legend = plt.legend(loc='upper right', bbox_to_anchor=(-0.35, 1.02), fontsize=12)

# # 加粗我们的方法名
# for text in legend.get_texts():
#     if 'our' in text.get_text().lower():
#         text.set_fontweight('bold')

# 图本身不被 tight_layout 挤压，用手动布局
fig.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.12)

# 设置 legend 放在左边外部（不缩小图）
legend = ax.legend(loc='center left',
                   bbox_to_anchor=(-0.05, 0.55),
                   bbox_transform=fig.transFigure,
                   fontsize=13)
# 加粗我们方法名
for text in legend.get_texts():
    if 'mobi' in text.get_text().lower():
        text.set_fontweight('bold')

ax.set_theta_offset(np.pi / 2)  # 让第一个轴从正上方开始
ax.set_theta_direction(-1)      # 逆时针方向排布
fig.subplots_adjust(top=0.92, bottom=0.50)
plt.tight_layout()
plt.savefig("final_radar_zsre.pdf", dpi=300, bbox_inches='tight')
plt.show()