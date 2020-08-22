import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import os
import re 
import sys
arg1 = sys.argv[1]
from collections import defaultdict
from utils.process import process,extract_mse
from scipy import stats
import pandas as pd
data_name=arg1
baseline=["DLCM_rever","DLCM_init","DLCM_rand","SetRank","DNN","DNN_naive"]
propensity=np.array([0.68, 0.61, 0.48, 0.34, 0.28, 0.2, 0.11, 0.1, 0.08, 0.06])
data=np.load("result_log/"+data_name+"_DNN/eta_1.0/dla_e/train_log_raw.npz")
dla_mse=extract_mse(data,propensity)
Yahoo_data=process("result_log/"+data_name+"_DNN/eta_1.0")[0]["eta_1.0"]["dla_e"]
Yahoo_baseline=["result_log/"+data_name+"_"+i+"/eta_1.0" for i in baseline]
metrics=["err_3","ndcg_3","err_10","ndcg_10"]
results=[]
for ind,method in enumerate(Yahoo_baseline):
    print(method)
    data=process(method)[0]["eta_1.0"]["dla_e"]
    results_cur=[]
    for metric in metrics:
        data_cur=np.array(data[metric])
        data_base=np.array(Yahoo_data[metric])
#         print(data_cur,data_base)
        label="+" if np.mean(data_cur)>np.mean(data_base) else "-"
        size=np.min([data_cur.shape[0],data_base.shape[0]])
        p=stats.ttest_rel(data_cur[:size],data_base[:size])[1]
        result_cur_metric="{:#.3g}".format(np.mean(data_cur))
        if p<0.05:
            result_cur_metric+=label
#         print(np.mean(data_cur),p)
        results_cur.append(result_cur_metric)
    if "naive" not in method:
        data_mse=np.load(method+"/dla_e/train_log_raw.npz")
        mse=extract_mse(data_mse,propensity,dla_mse)
        results_cur.append(mse)
    else:
        results_cur.append("-")
    results.append(results_cur)
dfObj = pd.DataFrame(results, columns = metrics+["$MSE_{propen}$"], index=baseline)
dfObj.to_csv("plots/"+data_name+"_table.csv")
fig, ax = plt.subplots()
marker=["o","s","v","*","^",">","<"]
for ind,method in enumerate(Yahoo_baseline):
    data=np.load(method+"/dla_e/valid_log_raw.npz")
    steps=data["steps"]
    ndcg_mean=np.mean(data["ndcg_10"],1)
    ndcg_std=np.std(data["ndcg_10"],1)
    ax.errorbar(steps,ndcg_mean,ndcg_std,marker=marker[ind],label=baseline[ind])
# plt.xticks(np.arange(min(ind1), max(ind1)+1000, 1000))
leg = ax.legend(bbox_to_anchor=(0.33, 0.0, 0.5, 0.5),loc=4,ncol=2)
plt.xlabel('Training steps', fontsize=10)
plt.ylabel('nDCG@10', fontsize=10)
ax.yaxis.grid()
ax.xaxis.grid()
# plt.savefig("/raid/taoyang/research/research_everyday/homework/intro_to_ir/projects/experiment_log/pic/Istella_comparison.pdf",bbox_inches='tight')
plt.savefig("plots/"+data_name+"_comparison.pdf",bbox_inches='tight', dpi = 600)
