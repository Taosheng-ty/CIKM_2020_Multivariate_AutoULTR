import argparse
class ULTRA_parser(argparse.ArgumentParser):
    def __init__(self,description=None):
        super(ULTRA_parser,self).__init__(description=description)
        self._add('-a','--algorithm',default="-1",dest="algorithm",help="this is algorithm id")
        self._add('--json_file',default="N",dest="json_file",help="this is algorithm id")
        self._add('-G','--GPU',default=0,type=int,dest="GPU",help="specify the gpu you are going to use")
        self._add('-o','--offline',default=False,action='store_true',dest="offline",help="whether online or not?")
        self._add('--toy',default=False,action='store_true',dest="toy",help="whether online or not?")
        self._add('--batch_size',default=256,type=int,dest="batch_size",help="whether online or not?")
        self._add('--lr',default=0.005,type=float,dest="lr",help="learning rate")
        self._add('--iteration',default=4000,type=int,dest="iteration",help="max_train_iteration")
        self._add('--result_folder',default="/home/taoyang/research/research_everyday/test_lab/ULTRA/ULTRA_implementation_logs/",dest="result_folder",help="the folder you are going to extract results. default is ULTRA/ULTRA_trial/logs/")
        self._add('--sub_folder',default="N",dest="sub_folder",help="the folder after result_folder")
    def _add(self,*arg,**kwargs):
        super(ULTRA_parser,self).add_argument(*arg,**kwargs)
    def parse_args(self,*arg,**kwargs):
        return super(ULTRA_parser,self).parse_args(*arg,**kwargs)
    def value(self):
        return vars(self.parse_args())
if __name__=="__main__":
        arg=ULTRA_parser()
        print(arg.value())