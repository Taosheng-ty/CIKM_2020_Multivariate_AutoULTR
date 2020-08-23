import os
import json
import sys
import re
sys.path.append("../../")
import  shutil
from datetime import datetime
import glob
from utils.ULTRA_parser import ULTRA_parser
# from utils.extract_eval import extract_eval
arg=ULTRA_parser()
aa=arg.parse_args()
# json_file
import json
from argparse import Namespace
gg=json.load(open(aa.json_file))
steps="steps_per_checkpoint" in gg
if not steps:
    gg["steps_per_checkpoint"]=200
argument=Namespace(**gg)

print(argument)
model_path=argument.model_path
out_path=argument.out_path

data_path=argument.data_path
iteration=10000

def write_slurm(path,cmd):
    fout = open(path, 'w')
    # set slurm parameters
#     fout.write('#!/bin/bash\n')
#     fout.write('#\n')
#     fout.write('#SBATCH --job-name='+job_name+'\n')
# #     fout.write('#SBATCH --partition=titan-long    # Partition to submit to \n')
# #     fout.write('#SBATCH --partition=titan-short    # Partition to submit to \n')
#     fout.write('#SBATCH --partition=debug    # Partition to submit to \n')
# #     fout.write('#SBATCH --gres=gpu:1\n')
#     fout.write('#SBATCH --ntasks=1\n')
#     fout.write("#SBATCH --tasks-per-node=8\n")
#     fout.write('#SBATCH --mem=5000    # Memory in MB per cpu allocated\n\n')
    for i in cmd:
        fout.write(i+"\n\n")
#     fout.write('exit\n')
    fout.close()

try:
    iteration=argument.iteration
except:
    pass
on_or_off="online" if argument.offline==False else "offline"
print(on_or_off,"on_or_off")
#print(hh)
eta=[1.0,0.2,0.4,0.6,0.8,1.2,1.4,1.6,1.8,2.0]
try:
    if argument.eta!=-1:
        eta=argument.eta
except:
    pass
click_model="pbm"
try:
    click_model=argument.click_model
except:
    pass
# eta=[0.2]
algorithm=["pdgd_exp_settings.json",
    "ipw_rank_exp_settings.json",
"naive_algorithm_exp_settings.json",
"pairwise_debias_exp_settings.json",
"dla_exp_settings.json",
"regression_EM_exp_settings.json",
"dbgd_exp_settings.json"]
algo_str=argument.algorithm
# print(type(algo_str),algo_str)
algo_id=list(algo_str.split(" "))

if algo_id[0]=="-":
    algo_id=[0,1,2,3,4,5,6]
else:
    algo_id=[int(i) for i in algo_id]
# print(algo_id,"algo_id",type(algo_id),"type")
algo=[algorithm[i] for i in algo_id]
# print(algo)
json_path=model_path+'example/'+on_or_off+'_setting/'
description=argument.description
# folder_name=description+"_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
folder_name=description
gg["sub_folder"]=folder_name
with open(aa.json_file,"w") as setting_file:
    json.dump(gg,setting_file)

clickmodel_path=model_path+'example/'+"ClickModel/"
times_to_run=1
if hasattr(argument,"times_to_run"):
                times_to_run=argument.times_to_run
python_env="ultra_p36"
if hasattr(argument,"python_env"):
                python_env=argument.python_env
        
def process_checkout(f,iteration):
        if not os.path.exists(f):
            return False
        with open (f, "r") as myfile:
                data=myfile.readlines()
        data_str=""
        for mm in data:
            data_str+=mm 
        if "global step "+str(iteration) in data_str: 
            return True
        else:
            return False
python_exe="~/miniconda3/envs/"+python_env+"/bin/python " if python_env else "python "

for i in range(len(eta)): 
    for time in range(times_to_run):
        for j in range(len(algo)):
            algo_name=algo[j][:5]
#             if not os.path.exists(json_path+"eta_file/"):
#                 os.mkdir(json_path+"eta_file/")
#             if not os.path.exists(clickmodel_path+click_model+"_0.1_1.0_4_"+str(eta[i])+".json"):
            time_str=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with open(clickmodel_path+click_model+"_0.1_1.0_4_1.0.json","r") as modelfile:
                model=json.load(modelfile)
            model["eta"]=eta[i]
#             model["atten"]=-1
            if hasattr(argument,"atten"):
                model["atten"]=argument.atten
#             with open(clickmodel_path+click_model+\
#                       "_0.1_1.0_4_"+str(eta[i])+".json","w") as modelfile:
#                 pass  
            with open(json_path+algo[j],"r") as readfile:
                exp1=json.load(readfile)
                exp1["selection_bias_cutoff"]=10
                if hasattr(argument,"ranking_model_hparams"):
                    exp1['ranking_model_hparams']=argument.ranking_model_hparams
                if hasattr(argument,"ranking_model"):
                    exp1['ranking_model']=argument.ranking_model
                if hasattr(argument,"train_input_feed"):
                    exp1['train_input_feed']=argument.train_input_feed
                if hasattr(argument,"valid_input_feed"):
                    exp1['valid_input_feed']=argument.valid_input_feed
                    exp1['test_input_feed']=argument.valid_input_feed
                if hasattr(argument,"train_list_cutoff"):
                    exp1["train_list_cutoff"]=argument.train_list_cutoff
                if hasattr(argument,"selection_bias_cutoff"):
                    exp1["selection_bias_cutoff"]=argument.selection_bias_cutoff
                if hasattr(argument,"metrics_topn"):
                    exp1["metrics_topn"]=argument.metrics_topn
                exp1["metrics"]=argument.metrics
                input_mode="Offline"
                if "Sto" in exp1['train_input_feed']:
                    input_mode="Sto"
                if "Det" in exp1['train_input_feed']:
                    input_mode="Det"
                job_name=input_mode+"_"+algo_name+"_eta_"+str(eta[i])+"_"+str(time)
                output_folder=out_path+folder_name+'/eta_'+str(eta[i])+"/"+algo_name+"/"+job_name
                output_folder=output_folder.replace(",","_")  
                if os.path.isdir(output_folder):
                    shutil.rmtree(output_folder)
                exp1["objective_metric"]="ndcg_10"
                exp1['train_input_hparams']="click_model_json="+output_folder+"/"+click_model+"_0.1_1.0_4_eta_"+str(eta[i])+".json"
#                 exp1['test_input_hparams']="click_model_json=./example/ClickModel/"+click_model+"_0.1_1.0_4_eta="+str(eta[i])+"_"+time_str+".json"     
#                 exp1['valid_input_hparams']="click_model_json=./example/ClickModel/"+click_model+"_0.1_1.0_4_eta="+str(eta[i])+"_"+time_str+".json"   
                exp1['learning_algorithm_hparams']="learning_rate="+str(argument.lr)
                lr=str(argument.lr)
                if hasattr(argument,'learning_algorithm_hparams'):
                    exp1['learning_algorithm_hparams']=argument.learning_algorithm_hparams
                    lr=argument.learning_algorithm_hparams
            print("******************************************","eta"+\
                  str(eta[i])+"algo"+algo[j]+str(j),"******************************************")
            
           
#             current_json=json_path+'eta_file/eta.json'+time_str+algo_name
            current_json=output_folder+"/setting.json"
            if os.path.exists(output_folder) ==False:
                os.makedirs(output_folder,exist_ok=True)
            with open(current_json, 'w') as outfile:
                json.dump(exp1, outfile)
            with open(output_folder+"/"+click_model+"_0.1_1.0_4_eta_"+str(eta[i])+".json","w") as modelfile:
                json.dump(model, modelfile)
            print(exp1)
            
            output_result=out_path+folder_name+'/eta_'+str(eta[i])+"/"
            output_result=output_result.replace(",","_")
            model_dir=output_folder+"/tmp_model/"
            output_dir=output_folder+"/tmp_output/"

            
            main_file=model_path+"main.py"
            tmp_data=data_path+'tmp_data/'
#             if argument.toy==True:
#                 tmp_data=data_path+'tmp_data_toy/'


            if os.path.exists(model_dir) ==False:
                os.makedirs(model_dir,exist_ok=True)
            if os.path.exists(output_dir) ==False:
                os.makedirs(output_dir,exist_ok=True)
#             sys.stdout=Logger(output_folder+"/","log.print")
            log_txt=output_folder+"/log.txt"
            log_eval=output_folder+"/eval.log"
            cmd=[]
#             cmd+=["cd "+model_path]
#             cmd+=[" cat "+aa.json_file]
#             cmd=cmd+ ["cd "+model_path+" \n\n cat "+current_json]
            cmd+=['export CUDA_VISIBLE_DEVICES='+str(argument.GPU)]
            if eta[i]==1.0:
                test_while_train="True"
            else:
                test_while_train="False"
            if hasattr(argument,"test_while_train") and argument.eta!=-1:
                test_while_train=argument.test_while_train
            cmd+=[python_exe+main_file+' --max_train_iteration='+str(iteration)+" --steps_per_checkpoint=" +str(gg["steps_per_checkpoint"])+' --data_dir='+tmp_data+" --batch_size "+str(argument.batch_size)+' --model_dir='+model_dir+' --test_while_train='+test_while_train+' --output_dir='+output_dir+' --setting_file='+current_json] 
#             fout = open(output_folder + '/'+job_name+'.sh', 'w')
            print(cmd)
            
#             return_code=os.system(cmd)
#             fout.write(cmd+"\n\n")
#             flag=process_checkout(log_txt,iteration)
            flag=True
            run_code_path="./"
            if flag:
                
                cmd+=[python_exe+main_file+' --data_dir='+tmp_data+' --model_dir='+model_dir+' --output_dir='+output_dir+' --test_only=True --setting_file='+current_json]
#                 return_code=os.system(cmd)
                cmd+=[python_exe +run_code_path+"utils/extract_eval.py "+ output_folder]
#                 cmdextract_eval()
#                 fout.write(cmd+"\n\n")
#                 extract_eval(output_folder)
#             return_code=os.system("echo "+ cmd+ " >> "+log_txt)
#             return_code=os.system(cmd)
#             os.remove(current_json)
    #         print("return code",return_code)
        
            cmd+=[python_exe+run_code_path+"utils/output_resuts.py --result_folder="+output_result]
    #         fout.write(cmd_output+"\n\n")
            write_slurm(output_folder + '/'+job_name+'.sh',cmd)
            command = ' chmod +x ' + output_folder + '/'+job_name+'.sh \n'
            command += ' bash ' + output_folder + '/'+job_name+'.sh'
            command += ' 2> '+output_folder+"/"+job_name+'.e 1> '+output_folder+"/"+job_name+'.o'
            
            print(command)
#             command=""
#             for each_cmd in cmd:
#                 command+=each_cmd+ " \n"
            os.system(command)
#         return_code=os.system(cmd_output)