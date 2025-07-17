import re
import matplotlib.pyplot as plt

input_file = 'delta_comp.log'   # 你的原始数据文件名
diff_file = 'diff_delta.txt'
plot_file = 'diff_delta.png'

diffs = []

with open(input_file, 'r', encoding='utf-8') as fin, open(diff_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        m = re.search(r'([-+]?\d+\.\d+)$', line.strip())
        if m:
            diff = float(m.group(1))
            diffs.append(diff)
            fout.write(f"{diff}\n")

# 删除前1000行
offset = 1000
diffs_slice = diffs[offset:]

# 让横坐标从1000开始
x_vals = list(range(offset, offset + len(diffs_slice)))

plt.figure(figsize=(10,4))
plt.plot(x_vals, diffs_slice, linewidth=0.6, color='b')
plt.title('Line-wise Diff Curve (Skip First 1000)')
plt.xlabel('Line Index')
plt.ylabel('Difference')
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_file, dpi=200)
plt.show()

print(f'剩余{len(diffs_slice)}个差值已绘制，曲线图保存为{plot_file}。横坐标从{offset}开始。')