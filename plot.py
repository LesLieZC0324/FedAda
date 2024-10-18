# Test Acc
# Fashion-MNIST IID_date IID_setting

import os
import pandas as pd
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

file_dict = 'results_new\IID_data_IID_net'
file_path1 = os.path.join(file_dict, 'fmnist_mnistCNN_400_iid[1]_K1[50]_K2[1]_A[2]_CB[0.0]_MC[1.0]_MB[0.0]_P[0.6].xlsx')
file_path2 = os.path.join(file_dict, 'fmnist_mnistCNN_200_iid[1]_K1[6]_K2[10]_A[0]_CB[0.0]_MC[1.0]_MB[0.0]_P[0.6].xlsx')
file_path3 = os.path.join(file_dict, 'fmnist_mnistCNN_200_iid[1]_K1[10]_K2[6]_A[0]_CB[0.0]_MC[1.0]_MB[0.0]_P[0.6].xlsx')
file_path4 = os.path.join(file_dict, 'fmnist_mnistCNN_50_iid[1]_K1[5]_K2[50]_A[0]_CB[0.0]_MC[1.0]_MB[0.0]_P[0.6].xlsx')
file_path5 = os.path.join(file_dict, 'fmnist_mnistCNN_10000_iid[1]_K1[1]_K2[1]_A[0]_CB[0.0]_MC[1.0]_MB[0.0]_P[0.6].xlsx')

data1 = pd.read_excel(file_path1)[0:400:4]
data2 = pd.read_excel(file_path2)[0:200:2]
data3 = pd.read_excel(file_path3)[0:200:2]
data4 = pd.read_excel(file_path4)
# data5 = pd.read_excel(file_path5)
data5 = pd.read_excel(file_path5)[0:10000:50]

bax = brokenaxes(xlims=((0, 20), (90, 100)),  #设置x轴裂口范围
                 # ylims=((0.2, 0.3), (0.7, 0.95)), #设置y轴裂口范围
                 hspace=0.25,  #y轴裂口宽度
                 wspace=0.25,  #x轴裂口宽度
                 despine=False,  #是否y轴只显示一个裂口
                 diag_color='r',  #裂口斜线颜色
                 )

bax.plot(data1['Completion Time'] / 1000, data1['Test Accuracy'], color='#df7a5e', linewidth=3, label='FedAda', marker="s",  markersize=8, markevery=9)
# plt.plot(data1['train_time'] / 1000, data1['test_acc'], color='#f27970', marker="s",  markersize=8)

bax.plot(data2['Completion Time'] / 1000, data2['Test Accuracy'], color='#3c405b', linewidth=3, label='HierFAVG', marker="o",  markersize=8, markevery=9)

# plt.plot(data3['Completion Time'][0:temp3[5]:2] / 1000, data3['Test Accuracy'][:temp3[5]:2], color='#05b9e2', linewidth=3, label='HierFAVG10', marker="^", markersize=8, markevery=2)

bax.plot(data4['Completion Time'] / 1000, data4['Test Accuracy'], color='#82b29a', linewidth=3, label='HFL', marker="v", markersize=8, markevery=7)

bax.plot(data5['Completion Time'][:199] / 1000, data5['Test Accuracy'][:199], color='#f2cc8e', linewidth=3, label='RAF', marker="D", markersize=8, markevery=9)

ax = plt.gca()

bax.set_xlabel('Time (×1000s)', fontproperties='Times New Roman', fontsize=28, labelpad=35)
bax.set_ylabel('Test Accuracy', fontproperties='Times New Roman', fontsize=28, labelpad=50)

bax.axs[1].get_yaxis().get_offset_text().set(size=24)
for tick in bax.axs[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(24)
for i in range(2):
    for tick in bax.axs[i].get_xaxis().get_major_ticks():
        tick.label.set_fontsize(24)

# plt.xlabel('Time (×1000s)', fontproperties='Times New Roman', fontsize=24)
# plt.xticks([0,5,10,15,20],fontsize=20)
# plt.ylabel('Test Accuracy', fontproperties='Times New Roman', fontsize=24)
# plt.yticks(fontsize=20)
# plt.ylabel('Train Loss', fontsize=14)
# bax.legend(fontsize=16,edgecolor='black', handlelength=1.5, loc='lower right')
# bax.legend(fontsize=20, edgecolor='black', handlelength=1.5, labelspacing=0.1, columnspacing=0.5, handletextpad=0.5, loc='lower right', ncol=2)
bax.legend(bbox_to_anchor=(1.05, 1.2), fontsize=24, edgecolor='black', handlelength=1.2, labelspacing=0.1, columnspacing=0.2, handletextpad=0.2, loc=0, ncol=4, borderaxespad=0)
# plt.xlim(0, 100)
# plt.ylim(0.84, 0.93)

ax = plt.gca()
# 绘制缩放图
axins = ax.inset_axes([0.2, 0.15, 0.75, 0.5])

# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins.plot(data1['Completion Time'] / 1000, data1['Test Accuracy'], color='#df7a5e', linewidth=3, label='FedAda', marker="s",  markersize=8, markevery=9, alpha=1.0)
axins.plot(data2['Completion Time'] / 1000, data2['Test Accuracy'], color='#3c405b', linewidth=3, label='HierFAVG', marker="o",  markersize=8, markevery=9, alpha=1.0)
axins.plot(data4['Completion Time'] / 1000, data4['Test Accuracy'], color='#82b29a', linewidth=3, label='HFL', marker="v", markersize=8, markevery=7, alpha=1.0)
axins.plot(data5['Completion Time'][:199] / 1000, data5['Test Accuracy'][:199], color='#f2cc8e', linewidth=3, label='RAF', marker="D", markersize=8, markevery=9, alpha=1.0)

#画小图
x1, x2, y1, y2 = 10, 19, 0.88, 0.94
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
x_interval = 2
axins.set_xticks(range(x1, x2 + 1, x_interval))
axins.tick_params(axis='x', labelsize=20)
axins.tick_params(axis='y', labelsize=20)

# plt.show()
plt.savefig('plot/fmnist_convergence_iid_iid_new.png', dpi=500, bbox_inches='tight')