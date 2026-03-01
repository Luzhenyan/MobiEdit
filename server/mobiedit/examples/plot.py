# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # 原始数据
# data = {
#     'method': ['ROME', 'MEMIT', 'WISE', 'AhphaEdit', 'Our Method'],
#     'Edit Success': [92.63, 91.00, 100.00, 94.00, 80.12],
#     'Time': [67.07, 67.07, 146.77, 67.07, 25.51],
#     'Memory': [46.1352, 46.1352, 46.3032, 46.1352, 6.2],
#     'Energy': [221.93, 221.93, 485.62, 221.93, 18.72]
# }
# df = pd.DataFrame(data)

# # 效率指标：反归一化 & 映射到 [0, 100]
# def inverse_norm(x, min_val, max_val):
#     return (1 - (x - min_val) / (max_val - min_val)) * 100

# for col in ['Time', 'Memory', 'Energy']:
#     df[col + '_score'] = inverse_norm(df[col], df[col].min(), df[col].max())

# # 平均效率得分
# df['Efficiency Score'] = df[['Time_score', 'Memory_score', 'Energy_score']].mean(axis=1)

# # 可视化
# fig, ax = plt.subplots(figsize=(8, 6))

# colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:green']
# markers = ['o', 's', '^', 'd', '*']

# for i, row in df.iterrows():
#     method = row['method']
#     x = row['Edit Success']
#     y = row['Efficiency Score']
#     is_our = 'our' in method.lower()
    
#     ax.scatter(x, y,
#                color=colors[i],
#                marker=markers[i],
#                s=120 if is_our else 80,
#                label=method,
#                edgecolor='black' if is_our else 'none',
#                linewidth=1)
#     ax.annotate(method,
#                 (x + 0.5, y + 0.5),  # 文字位置偏移
#                 fontsize=10,
#                 color=colors[i],
#                 weight='bold' if is_our else 'normal'
#                )

# # 坐标轴 & 标尺
# ax.set_xlabel('Edit Success', fontsize=12, weight='bold')
# ax.set_ylabel('Efficiency Score', fontsize=12, weight='bold')
# ax.set_xlim(0, 105)
# ax.set_ylim(-5, 105)
# ax.grid(True, linestyle='--', linewidth=0.5)
# ax.axhline(50, color='gray', linewidth=0.6, linestyle='--')
# ax.axvline(85, color='gray', linewidth=0.6, linestyle='--')

# # 图标题 & 图例
# plt.title("Efficiency vs Edit Success Trade-off", fontsize=14, weight='bold')
# plt.legend(loc='lower left')

# # 保存 & 显示
# plt.tight_layout()
# plt.savefig("scatter_tradeoff_edit_efficiency.pdf", dpi=300, bbox_inches='tight')
# # plt.show()

import matplotlib.pyplot as plt

# ==== 数据 ====
methods = [
    'ROME',
    'zo',
    'zo + early stopping',
    'zo + prefix cache',
    'MobiEdit'
]

x = [92.6348, 86.88, 86.88, 82.68, 80.12]  # Edit Succ
y = [4024.36, 4443.99, 2931.14, 2320.88, 1530.79]  # Time(s)

colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:green', 'tab:red']
markers = ['*', 'o', 'o', 'o', 'o']
sizes = [100, 80, 80, 80, 90]

# ==== 画图 ====
plt.figure(figsize=(8, 6))
for i in range(len(methods)):
    plt.scatter(x[i], y[i], color=colors[i], marker=markers[i], s=sizes[i], zorder=3)
    plt.text(x[i]+0.5, y[i], methods[i], fontsize=22, va='center')

# ==== 路线箭头 ====
# zo → zo + early stop → zo + early stop + prefix cache
plt.annotate('', xy=(x[2], y[2]), xytext=(x[1], y[1]),
             arrowprops=dict(arrowstyle='->', color='orange', lw=1.8))

plt.annotate('', xy=(x[4], y[4]), xytext=(x[2], y[2]),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.8))

plt.annotate('', xy=(x[4], y[4]), xytext=(x[3], y[3]),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.8))

# 再一条 zo → zo + prefix cache
plt.annotate('', xy=(x[3], y[3]), xytext=(x[1], y[1]),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.8))

# ==== 设置轴 ====
plt.xlabel("Edit Success (%)", fontsize=24)
plt.ylabel("Time (s)", fontsize=24)
# plt.title("Ablation Study: Time vs Edit Success", fontsize=14)

plt.xlim(75, 95)
plt.ylim(1000, 4700)
plt.tick_params(axis='both', labelsize=20)
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("ablation_edit_success_vs_time.pdf", dpi=300)
plt.show()