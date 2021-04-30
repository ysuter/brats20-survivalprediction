#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

inpall = pd.read_csv("./BraTS2020/boxplotinp.csv")

orderlist = ['Age, shape, position', 'Age, shape, position, power transform', 'No prior', 'No prior, power transform', 'Robustness prior', 'Robustness prior, power transform', 'Sequence prior', 'Sequence prior, power transform', 'Both priors', 'Both priors, power transform transform']
orderlist_hue = ['Age, shape, position', 'No prior', 'Robustness prior', 'Sequence prior', 'Both priors']


colpal = sns.color_palette("inferno", len(orderlist_hue))
my_pal = {orderlist_hue[0]: colpal[0],
        orderlist_hue[1]: colpal[1],
        orderlist_hue[2]: colpal[2],
        orderlist_hue[3]: colpal[3],
        orderlist_hue[4]: colpal[4]}

colpal_full = sns.color_palette("inferno", len(orderlist))
my_pal_full = {elem: colpal_full[elemidx] for elemidx, elem in enumerate(orderlist)}

colpal_new = sns.color_palette("inferno", 12)

my_pal_new = {orderlist[0]: colpal_new[-9],
              orderlist[1]: colpal_new[-9],
            orderlist[2]: colpal_new[-7],
            orderlist[3]: colpal_new[-7],
            orderlist[4]: colpal_new[-5],
            orderlist[5]: colpal_new[-5],
            orderlist[6]: colpal_new[-3],
            orderlist[7]: colpal_new[-3],
            orderlist[8]: colpal_new[-1],
            orderlist[9]: colpal_new[-1]}

hatches = '//'
plt.rcParams.update({'hatch.color': 'white'})

circ1 = mpatches.Patch(facecolor='grey', edgecolor='k', label='No transform')
circ2 = mpatches.Patch(facecolor='grey', edgecolor='k', hatch=r'\\\\', label='Power transform')

plt.figure(figsize=(6, 9.2))
ax = sns.boxplot(data=inpall, x="Modelname", y="Balanced Accuracy", order=orderlist, palette=my_pal_new)
plt.ylim([0, 1])
# plt.xlabel("Model")
# plt.xticks(rotation=90)
plt.title("Balanced accuracy", fontweight='bold')
for i, box in enumerate(ax.artists):
    if ((i+1) % 2 == 0):
        box.set_hatch(hatches)
leg = ax.legend(handles=[circ1, circ2], loc="upper right", frameon=False, labelspacing=1.2)
for patch in leg.get_patches():
    patch.set_height(15)
    patch.set_y(-3)
plt.xticks(np.arange(1, 2*len(orderlist_hue)+1, step=2)-0.5, orderlist_hue)
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("./BraTS2020/balacc_boxplot.png")
plt.show()

plt.figure(figsize=(6, 8))
ax = sns.boxplot(data=inpall, x="Modelname", y="Accuracy", order=orderlist, palette=my_pal_new)
plt.ylim([0, 1])
# plt.xlabel("Model")
# plt.xticks(rotation=90)
plt.title("Accuracy", fontweight='bold')
for i, box in enumerate(ax.artists):
    if ((i+1) % 2 == 0):
        box.set_hatch(hatches)
leg = ax.legend(handles=[circ1, circ2], loc="upper right", frameon=False, labelspacing=1.2)
for patch in leg.get_patches():
    patch.set_height(15)
    patch.set_y(-3)
plt.xticks(np.arange(1, 2*len(orderlist_hue)+1, step=2)-0.5, orderlist_hue)
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("./BraTS2020/acc_boxplot.png")
plt.show()

plt.figure(figsize=(6, 8))
ax = sns.boxplot(data=inpall, x="Modelname", y="MSE", order=orderlist, palette=my_pal_new)
# plt.ylim([0, 1])
# plt.xlabel("Model")
# plt.xticks(rotation=90)
plt.title("Mean squared error", fontweight='bold')
for i, box in enumerate(ax.artists):
    if ((i+1) % 2 == 0):
        box.set_hatch(hatches)
leg = ax.legend(handles=[circ1, circ2], loc="upper right", frameon=False, labelspacing=1.2)
for patch in leg.get_patches():
    patch.set_height(15)
    patch.set_y(-3)
plt.ylabel(r'Mean squared error / days$^2$')
plt.xticks(np.arange(1, 2*len(orderlist_hue)+1, step=2)-0.5, orderlist_hue)
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("./BraTS2020/mse_boxplot.png")
plt.show()

plt.figure(figsize=(6, 8))
ax = sns.boxplot(data=inpall, x="Modelname", y="spearmanr", order=orderlist, palette=my_pal_new)
plt.ylim([-0.2, 0.7])
# plt.xlabel("Model")
# plt.xticks(rotation=90)
plt.title("Spearman's rho", fontweight='bold')
for i, box in enumerate(ax.artists):
    if ((i+1) % 2 == 0):
        box.set_hatch(hatches)
leg = ax.legend(handles=[circ1, circ2], loc="lower right", frameon=False, labelspacing=1.2)
for patch in leg.get_patches():
    patch.set_height(15)
    patch.set_y(-3)
plt.ylabel("Spearman's rho")
plt.xticks(np.arange(1, 2*len(orderlist_hue)+1, step=2)-0.5, orderlist_hue)
plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=13)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("./BraTS2020/spearman_boxplot.png")
plt.show()
