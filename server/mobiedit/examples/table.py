import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# --------- 数据定义 ----------
data = {
    'Method': ['ROME', 'MEMIT', 'WISE', 'AhphaEdit', 'Our Method'],
    'Memory(GB)': [46.14, 46.14, 46.30, 46.14, 6.20],
    'K60_Time': [4543.78, 4543.78, 11359.44, 4543.78, 1902.88],
    'K60_Energy': [0.25, 0.25, 0.63, 0.25, 0.02],
    'K70_Time': [4276.49, 4276.49, 8552.99, 4276.49, 1477.67],
    'K70_Energy': [0.24, 0.24, 0.47, 0.24, 0.02],
    'OnePlus_Time': [3252.81, 3252.81, 6505.63, 3252.81, 1211.83],
    'OnePlus_Energy': [0.18, 0.18, 0.36, 0.18, 0.01]
}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- 可调参数 ----------
col_width = 1.2
row_height = 0.6

n_data_rows = 0  # 暂时不画数据
n_header_rows = 2
n_total_rows = n_data_rows + n_header_rows
n_columns = 8  # Method + Memory + 6 子列(Time/Energy for 3 devices)

fig_width = col_width * n_columns
fig_height = row_height * n_header_rows

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_xlim(0, n_columns)
ax.set_ylim(0, n_header_rows)
ax.axis('off')

# ---------- 分组信息 ----------
columns = ['Method', 'Memory(GB)',
           'K60_Time', 'K60_Energy',
           'K70_Time', 'K70_Energy',
           'OnePlus_Time', 'OnePlus_Energy']

group_headers = ['', '', 'K60', 'K60', 'K70', 'K70', 'OnePlus', 'OnePlus']
sub_headers = ['Method', 'Memory(GB)', 'Time', 'Energy', 'Time', 'Energy', 'Time', 'Energy']

# ---------- 背景格子 + 文本 ----------
for col_idx in range(n_columns):
    for row_idx in range(n_header_rows):
        x = col_idx
        y = n_total_rows - row_idx - 1

        # 样式格子
        ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='white', linewidth=0.8))

        # 第1行：分组表头
        if row_idx == 0:
            if col_idx == 0:
                ax.text(x + 0.5, y + 0.5, 'Method', ha='center', va='center', fontsize=10, fontweight='bold')
            elif col_idx == 1:
                ax.text(x + 0.5, y + 0.5, 'Memory(GB)', ha='center', va='center', fontsize=10, fontweight='bold')
            elif col_idx in [2, 4, 6]:
                label = group_headers[col_idx]
                ax.text(x + 1.0, y + 0.5, label, ha='center', va='center', fontsize=10, fontweight='bold')

        # 第2行：子表头
        if row_idx == 1:
            label = sub_headers[col_idx]
            ax.text(x + 0.5, y + 0.5, label, ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("zsre_table.pdf", dpi=300)
plt.show()

