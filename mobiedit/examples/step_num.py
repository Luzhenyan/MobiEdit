import matplotlib.pyplot as plt
import numpy as np

bins = ['<100', '100-200', '200-300', '300-400', '400-500', '500-600']
zsre = np.array([0.104166667, 0.270833333, 0.020833333, 0.166666667, 0.0625, 0.375])
cf   = np.array([0.055555556, 0.203703704, 0.018518519, 0.240740741, 0.12962963, 0.351851852])

zsre_cum = np.cumsum(zsre)
cf_cum = np.cumsum(cf)

bar_width = 0.35
x = np.arange(len(bins))

fig, ax1 = plt.subplots(figsize=(9,5))

# 柱状图
ax1.bar(x - bar_width/2, zsre, width=bar_width, color='tab:blue', alpha=0.7, label='ZsRE')
ax1.bar(x + bar_width/2, cf,   width=bar_width, color='tab:orange', alpha=0.7, label='CounterFact')
ax1.set_ylabel('Proportion in Step Range', fontsize=22)
ax1.set_xlabel('Step', fontsize=22)
# ax1.tick_params(axis='y', labelsize=18) 
ax1.set_xticklabels(bins, fontsize=15)

# 累计折线
ax2 = ax1.twinx()
line1, = ax2.plot(x, zsre_cum, color='tab:blue', marker='o', linewidth=2)
line2, = ax2.plot(x, cf_cum,   color='tab:orange', marker='o', linewidth=2)
ax2.set_ylabel('Cumulative Proportion', fontsize=22)
# ax2.tick_params(axis='y', labelsize=18) 
ax2.set_ylim(0,1.08)

# 标注累计点数据
for i in range(len(x)):
    ax2.text(x[i], zsre_cum[i]+0.025, f'{zsre_cum[i]:.2f}', color='tab:blue', fontsize=22, ha='center')
    ax2.text(x[i], cf_cum[i]-0.055, f'{cf_cum[i]:.2f}', color='tab:orange', fontsize=22, ha='center')

# 合并图例
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1+handles2, labels1+labels2, loc='upper left', fontsize=22)

# plt.title('Step-to-Convergence: Interval Distribution and Cumulative', fontsize=14)
plt.tight_layout()
plt.savefig('fig_step_num.pdf', dpi=300)
plt.show()