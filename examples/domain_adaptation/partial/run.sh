# 
# Office-Home 
CUDA_VISIBLE_DEVICES=0 python source_only.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 5 -i 500 --seed 0 --log logs/officehome_src_only/OfficeHome_Ar2Cl  
CUDA_VISIBLE_DEVICES=0 python source_only_ctr.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --src-ctr-factor 0.1 --cluster-momentum 0.99 --pretrain logs/officehome_src_only/ --epochs 1 -i 500 --seed 0 --log logs/officehome_src_only_ctr/OfficeHome_Ar2Cl  
CUDA_VISIBLE_DEVICES=0 python cwan.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --num-cluster 65 --mixing-factor 0.5 --factor1 0 --factor2 0.1 --factor3 1 --factor4 0.1 --cluster-momentum 0.99 --pretrain logs/officehome_src_only_ctr/ --epochs 30 --seed 0 --log logs/officehome_cwan/OfficeHome_Ar2Cl  

# 
# Office-31 
CUDA_VISIBLE_DEVICES=0 python source_only.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 1 -i 500 --seed 0 --log logs/office31_src_only/Office31_A2D 
CUDA_VISIBLE_DEVICES=0 python cwan.py data/office31 -d Office31 -s A -t D -a resnet50 --num-cluster 31 --mixing-factor 0.5 --factor1 0.5 --factor2 0.5 --factor3 1 --factor4 0.1 --cluster-momentum 0.9 --pretrain logs/office31_src_only/ --epochs 30 --i 100 --seed 0 --log logs/office31_cwan/Office31_A2D  

# VisDA-2017
CUDA_VISIBLE_DEVICES=0 python source_only.py data/visda-2017 -d VisDA2017 -s Real -t Synthetic -a resnet50 --epochs 1 --seed 0 -i 500 --log logs/VisDA_src_only/VisDA2017_R2S --per-class-eval --center-crop 
CUDA_VISIBLE_DEVICES=0 python cwan.py data/visda-2017 -d VisDA2017 -s Real -t Synthetic -a resnet50 --num-cluster 12 --mixing-factor 0.5 --factor1 0.5 --factor2 0.5 --factor3 1 --factor4 0.1 --cluster-momentum 0.9 --pretrain logs/VisDA_src_only/ --epochs 30 --seed 0 --log logs/VisDA_cwan/VisDA2017_R2S --per-class-eval --center-crop 

