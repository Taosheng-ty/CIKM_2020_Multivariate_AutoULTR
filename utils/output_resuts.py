import os
import re
import pandas as pd
import tensorflow as tf
import sys
import json
import os
import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from datetime import datetime
from argparse import Namespace
# sys.path.append("/home/taoyang/research/research_everyday/test_lab/ULTRA/")
from datetime import datetime
from ULTRA_parser import ULTRA_parser
from datetime import datetime
arg=ULTRA_parser()
aa=arg.parse_args()

if aa.json_file=="N":
#     sub=os.listdir(director)[-1]
    director=aa.result_folder+"/"
else:
    gg=json.load(open(aa.json_file))
    argument=Namespace(**gg)
    director=argument.result_folder+argument.sub_folder+"/"
import numpy as np
ranklist=10
all_algo=os.listdir(director)
algo=[]


def tabulate_events(dpath,dname,out = defaultdict(list)):
#     print(dpath)
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload()]
    
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

#     out = defaultdict(list)
    flag=False
    if len(out.keys())>0:
        flag=True
    steps = []
#     if out
    for tag in tags:
        steps = [[e.step] for e in summary_iterators[0].Scalars(tag)]
#         print(steps)
        if len(steps)<2:
            steps=-1
            return out, steps
        i=0
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
#             if len(set(e.step for e in events)) != 1:
#                 steps=-1
#                 return out, steps
            if not flag:
                out[tag].append([e.value for e in events])
            else:
                out[tag][i].append([e.value for e in events][0])
            i+=1
#             print([e.value for e in events],"dddd")
#     out["step"]
    return out, steps



def average(input_dict):
    aver_dict = copy.deepcopy(input_dict)
    aver_std_dict = copy.deepcopy(input_dict)
    numpy_dict= copy.deepcopy(input_dict)
    for key,value in input_dict.items():
        ind=0
        numpy_dict[key]=np.array(aver_std_dict[key])
        for i in value:            
            mean = sum(i)/len(i)
            variance = sum([((x - mean) ** 2) for x in i]) / len(i)
            stddev = variance ** 0.5
            aver_std_dict[key][ind]=[mean,stddev]
            aver_dict[key][ind]=mean
            ind+=1
    return aver_dict,aver_std_dict,numpy_dict

time_str=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
parent_path=director
data_set=["test_log","valid_log","train_log"]
algo_path=[parent_path+path_next for path_next in os.listdir(parent_path) if "."not in path_next]
for path in algo_path:
    for data_set_name in data_set:
#         print(data_set_name,"data_set_name",path)
        out = defaultdict(list)
        global_step=-1
        for (dirpath, dirnames, filenames) in os.walk(path):
            flag=False
            
            for filename in filenames:
#                 print(dirpath, dirnames, filename)
                if filenames!=[] and "tfevents" in filename and data_set_name in dirpath:
                    print(dirpath, dirnames, filename)
#                     out = defaultdict(list)
                    steps=-1
                    out, steps=tabulate_events(dirpath, filename,out)
#                     print(dirpath, filename,steps)
                    if steps==-1:
    #                     print(steps,"steps")
                        continue
                    if steps!=-1:
                        global_step=steps
        if global_step==-1:
            continue
        out["steps"]=global_step
        df = pd.DataFrame(out)
    #     path=dirpath.split("/")[1]
#                     print(dirpath, dirnames, filename)
        df.to_csv(path+'/'+data_set_name+".csv")
#         print(out)
        aver_dict,aver_std_dict,numpy_dict=average(out) 
        df_aver_dict = pd.DataFrame(aver_dict)
#         df_aver_dict.to_csv(path+'/'+data_set_name+time_str+"_aver.csv")
#         df_aver_std_dict = pd.DataFrame(aver_std_dict)
#         df_aver_std_dict.to_csv(path+'/'+data_set_name+time_str+"_aver_std.csv")
#         np.savez(path+'/'+data_set_name+time_str+"_raw.npz", **numpy_dict)
        df_aver_dict.to_csv(path+'/'+data_set_name+"_aver.csv")
        df_aver_std_dict = pd.DataFrame(aver_std_dict)
        df_aver_std_dict.to_csv(path+'/'+data_set_name+"_aver_std.csv")
        np.savez(path+'/'+data_set_name+"_raw.npz", **numpy_dict)
#         print(out)








# for i in range(len(all_algo)):
#     if len(all_algo[i])<6:
#         algo.append(all_algo[i])

# data={}
# # algo=["onlinenaiv","onlinedla_","onlineipw_","onlineregr","onlinepair","onlinepdgd","onlinedbgd"]

# data["algo"]=["onlinenaive","onlinedla_e","onlineipw_r","onlineregre","onlinepairw","onlinepdgd_","onlinedbgd_","onlinedbgdd"]
# name_length=11
# if argument.offline ==True:
#     name_length=12
#     data["algo"]=["offlinenaive","offlinedla_e","offlineipw_r","offlineregre","offlinepairw","offlinepdgd_","offlinedbgd_","offlinedbgdd","offlinedlat_","offlinedlac_","offlinedlaa_","offlinedla_D","offlinedla_a","offlinedla_S"]
# # every=200

# metrics=['err_1','err_3','err_5','err_10','ndcg_1','ndcg_3','ndcg_5','ndcg_10']
# eta=["0.2","0.4","0.6","0.8","1.0","1.2","1.4","1.6","1.8","2.0","3.0","5.0","7.0","9.0","10.0"]
# iteration=["1000","2000","3000","4000","5000","6000","7000","8000","9000","10000"]
# time_str=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# for i in range(len(iteration)):
#     data[iteration[i]]=[[]for i in range(len(data["algo"]))]
# for i in range(len(metrics)):
#     data[metrics[i]]=[[]for i in  range(len(data["algo"]))]
# for i in range(len(eta)):
#     data["ndcg_10_eta_"+eta[i]]=[[]for i in  range(len(data["algo"]))]
# for i in range(len(eta)):
#     data["mse_eta_"+eta[i]]=[[]for i in  range(len(data["algo"]))]
# # for i in range(len(eta)):
# #     data["ndcg_10_eta_"+eta[i]]=[[]for i in  range(len(algo))]
# algo_str=""    
# thousand_step=int(1000/argument.steps_per_checkpoint)    
# for i in range(len(algo)):
#     algo_str+="_"+algo[i]
#     algo_settings=os.listdir(director+algo[i])
# #     print(director,"director",algo[i],"algo"+str(algo[i]))
# #     print(algo_settings)
#     for j in algo_settings:
#         if "name_" in j:
#             eta_esti="mse_eta_"+j.split("eta_",1)[1][:3]
            
#             if eta_esti not in data.keys():
#                 data[eta_esti]=[[]for i in  range(len(algo))]
#     for j in algo_settings:
#         if "name_" in j:
#             print(j,algo[i],"j,algo[i]")
#             algo_sub=j.split("name_",1)[1][:name_length]
#             eta_sub="ndcg_10_eta_"+j.split("eta_",1)[1][:3]
#             eta_esti="mse_eta_"+j.split("eta_",1)[1][:3]
# #             print(j.split("name_",1)[1][:10])
# #             print(j.split("eta_",1)[1][:3])

#             algo_id=data["algo"].index(algo_sub)
# #             print(algo_id)
# #             print(eta_sub)
# #             if eta_sub not in data.keys():
# #                 data[eta_sub]=[[]for i in  range(len(algo))]
#             direc=director+algo[i]+"/"+j+'/tmp_model/valid_log/'
#             tf_log=direc+ os.listdir(direc)[0]
#             count=0
#             direc_val=director+algo[i]+"/"+j+'/tmp_model/valid_log/'
#             tf_log_val=direc_val+ os.listdir(direc_val)[0]
#             best_summary=None
#             best_ndcg=0
#             best_id=0
#             count_val=0
#             train_summary=None
#             for summary in tf.train.summary_iterator(tf_log_val):
                
# #                 print(count)
#                 for v in summary.summary.value:
#                     if v.tag =="ndcg_10":
#                         if v.simple_value>best_ndcg:
#                             best_ndcg=v.simple_value
# #                             best_summary=summary
#                             best_id=count_val            
#                 count_val=count_val+1

            
            
#             for summary in tf.train.summary_iterator(tf_log):
                
# #                 print(count)
#                 if count==best_id:
#                     best_summary=summary 
                
                
#                 if str(str(count*argument.steps_per_checkpoint)+eta_sub)  not in data.keys():
                    
#                     data[str(count*argument.steps_per_checkpoint)+eta_sub]=[[]for i in  range(len(data["algo"]))]
                
#                 for v in summary.summary.value:
#                     if v.tag =="ndcg_10":
#                         data[str(str(count*argument.steps_per_checkpoint)+eta_sub)][algo_id].append(v.simple_value)
# #                     data[str(count)][algo_id].append(v.simple_value)
#                 if count%thousand_step==0 and eta_sub[-3:]=="1.0":
                
#                     step_id=count//thousand_step-1
# #                     print(step_id)
#                     step=iteration[step_id]
# #                     print(step)
#                     for v in summary.summary.value:
#                         if v.tag =="ndcg_10":
#                             data[step][algo_id].append(v.simple_value)
#                 count=count+1
#             for v in best_summary.summary.value:

#                     if v.tag in metrics  and eta_sub[-3:]=="1.0":
#                         tag_id=metrics.index(v.tag)
#                         data[v.tag][algo_id].append(v.simple_value)
#                     if v.tag =="ndcg_10" :
# #                         tag_id=eta.index(eta_sub)
# #                         print(algo_id,"algo——id")
# #                         print(data[eta_sub])
#                         data[eta_sub][algo_id].append(v.simple_value)
#                     if v.tag=="click_likelyhood":
#                         if not data.get(eta_sub+"__"+v.tag):
#                             data[eta_sub+"__"+v.tag]=[[]for i in  range(len(data["algo"]))]
#                         data[eta_sub+"__"+v.tag][algo_id].append(v.simple_value)
#             direc=director+algo[i]+"/"+j+'/tmp_model/train_log/'
#             tf_log=direc+ os.listdir(direc)[0]
# #             print(tf_log,"tf_log")
#             count_train=0
# #             print("beast",best_id)
#             for summary in tf.train.summary_iterator(tf_log):
                
                
#                 if count_train==best_id*argument.steps_per_checkpoint:
# #                        print("generate new train summary")
#                        train_summary=summary 
#                 count_train=count_train+1
# #             print("count_train",count_train)
# #             train_summary=summary 
#             for v in train_summary.summary.value:
# #                 print(v.tag,v.simple_value)
#                 if v.tag[:-2] in "Inverse_Propensity_weights_Examination_Probability_t_plus_Probability_":
# #                             print(v.tag)
# #                             if v.tag[:-2] in "Examination_Probability_":
# #                                 print(v.tag)
# #                                 print(v.simple_value)
# #                             print(algo_sub,"algo_sub when eta",v.tag,"v.tag")
#                             data[eta_esti][algo_id].append(v.simple_value)

# pro_ground=np.array([0.68, 0.61, 0.48, 0.34, 0.28, 0.20, 0.11, 0.10, 0.08, 0.06])
# pro_ground=pro_ground/pro_ground[0]
# data_xl=pd.DataFrame.from_dict(data)
# data_xl.to_csv(director+"log_full"+time_str+algo_str+".csv")
# print("output csv file at "+director+"log_full"+time_str+algo_str+".csv")
# # for i in eta:
# # #     print(data["mse_eta_"+i])
# #     key_eta_esti="mse_eta_"+i
# #     for j in range(len(data["algo"])):
        
# #         if data[key_eta_esti][j]!=[]:
# # #             print(data[key_eta_esti][j],data["algo"],"debug")
# #             array=np.array(data[key_eta_esti][j])
# #             try:
# #                 array=np.reshape(array,(-1,10))
# #             except:
# #                 array=np.reshape(array,(-1,9))
# # #             array=np.mean(array,0)
# # #             print(np.power(pro_ground,float(i)),"pweo")
# # #             print(array)
# #             mse_eta=np.square(array-np.power(pro_ground,float(i))).mean(1)
# # #             mse_std=np.square(array-np.power(pro_ground,float(i))).std()
# #             if "reg" not in data["algo"][j] :
# # #                 print(data["algo"][j],"inverse")
# # #                 print(array)

# #                 mse_eta=(np.square(1/array-np.power(pro_ground,float(i)))).mean(1)
# # #                 mse_eta_std=(np.square(1/array-np.power(pro_ground,float(i)))).std()
# #             mse_mean=mse_eta.mean()
# #             mse_std=mse_eta.std()
# #             data[key_eta_esti][j]=[mse_mean,mse_std]
# #             print(type(mse_eta))
# # only_mean_data=data.copy()
# only_mean_data= {key: value[:] for key, value in data.items()}
# for i in data.keys():
#     if i !="algo":
#         for j in range(len(data[i])):
#             table_ij=data[i][j]
#             if len(table_ij)>0:
#                 data[i][j]=sum(table_ij)/len(table_ij)
#                 table_ij=np.array(table_ij)
#                 shape=table_ij.shape
#                 mean=table_ij.mean()
#                 std=table_ij.std()
                
#                 data[i][j]=[mean,std,shape]
# #                 print(data[i][j],[mean,std,shape],"[mean,std,shape] bef")
#                 only_mean_data[i][j]=mean
# #                 print(data[i][j],[mean,std,shape],"[mean,std,shape] after")
# data_xl=pd.DataFrame.from_dict(data)
# data_xl.to_csv(director+"log_mean_std"+time_str+algo_str+".csv")
# print("output csv file at "+director+"log_mean_std"+time_str+algo_str+".csv")
# data_xl=pd.DataFrame.from_dict(only_mean_data)
# data_xl.to_csv(director+"log_mean"+time_str+algo_str+".csv")
# print("output csv file at "+director+"log_mean"+time_str+algo_str+".csv")
    