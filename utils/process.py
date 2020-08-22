import scipy
import json
import os
import re 
import numpy as np
from collections import defaultdict
from scipy import stats
def merge(dict_total,dict_ind):
    for i in dict_ind.keys():
        dict_total[i].append(dict_ind[i])
def process(f,verbose=0):
    data={}
    stat={}
    if type(f)!=list:
        f=[f]
    for path in f:
        file=os.walk(path)
        for i,j,k in file:
            
            if len(k)>0 and "eval.json" in k:
                if verbose==1:
                    print(i,j,k)
                eta=None
                path_cur=0
                path_split=i.split("/")
                for m in i.split("/"):
                    z=re.match("(eta_\d*\.?\d*$)",m)
                    path_cur+=1
                    if z:
                        eta=z.groups()[0]
                        method=path_split[path_cur]
                        break
#                 print(eta)
                if not data.get(eta):
                    data[eta]=defaultdict(list)
                    stat[eta]=defaultdict(list)
                if not data.get(eta).get(method):
                    data[eta][method]=defaultdict(list)
                    stat[eta][method]=defaultdict(list)
                path=i+"/eval.json"
                with open(path) as json_file:
                    if verbose==1:
                        print(path)
                    data_eval = json.load(json_file)
                data_cur=data[eta][method]
                merge(data_cur,data_eval)
    for key_1 in data.keys():
            for key_2 in data[key_1].keys():
                for key_3 in data[key_1][key_2].keys():
                    array=np.array(data[key_1][key_2][key_3])
                    mean=float(format(np.mean(array), '.5f'))
                    std=float(format(np.std(array), '.5f'))
                    stat[key_1][key_2][key_3]=[mean,std]
    return data,stat
def extract_mse(data,propensity,p_test_base=np.array(100)):
    pro_esti=[]
    rank_count=[]
    for i in list(data.keys()):
        if "Inverse_Propensity_weights" in i: 
            rank_count.append(i) 
    propensity=propensity[:len(rank_count)]
    for i in range(len(rank_count)):
        pro_esti.append(data["Inverse_Propensity_weights_"+str(i)][-1])
    pro_esti=np.array(pro_esti)
#     print(pro_esti)
    propensity=propensity[0]/propensity
    propensity=propensity[:,np.newaxis]
    err=np.mean(np.square(pro_esti-propensity),0)
    if np.mean(p_test_base)!=100:
#         print(p_test_base,err)
        p_value=stats.ttest_rel(p_test_base,err)[1]
#         print(p_value,"p_value")
        label="+" if np.mean(err)<np.mean(p_test_base) else "-"
        value="{:#.3g}".format(np.mean(err))
        if p_value<0.05:
            value+=label
        return value
    else:
#     err=np.array(pro_esti-propensity[0]/propensity)
# #     print(np.square(err),"err")
#     mse=np.mean(np.square(err))
        return err