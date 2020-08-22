import re
import json
import sys
import os
def extract_eval(result_folder):
        for i in os.listdir(result_folder):
            if ".o" in i:
                output_file=i
        with open (result_folder+"/"+output_file, "r") as myfile:
                data=myfile.readlines()
        for i in data:
            o=re.findall('eval: (.*?)\n', i)
            if len(o)!=0:
                o=o[0]
                break
        dic_list=[]
        for x in o.split(" "):
            a,b=x.split(":")
            dic_list.append([a,float(b)])
        d = dict(dic_list)
        with open (result_folder+"/eval.json", "w") as myfile:
            json.dump(d,myfile)
result_folder=sys.argv[1]
extract_eval(result_folder)