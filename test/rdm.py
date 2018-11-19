import sys
sys.path.insert(0,'/home/scott/Documents/research/3_vqa/hqca/tools/')
import rdmf

wf = rdmf.wf_BD(1,0,0)
rdm2 = rdmf.build_2rdm(wf,rdmf.map_Lambda)
#print(rdm2)
#sf_rdm2 = rdmf.spin_free_rdm2(rdm2,rdmf.map_lambda)
#print(sf_rdm2)

