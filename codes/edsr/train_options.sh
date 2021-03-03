# -*- coding: utf-8 -*-
# DBPN

#10006번
python main.py --model DBPN --scale 2 --batch_size 20 --patch_size 64 --epochs 1000 --decay 500 --loss '1*MSE' --save DBPN_BI_X2_jpg_mse --save_results --reset --ext sep-reset 
#10002번
python main.py --model DBPN --scale 4 --batch_size 20 --patch_size 128 --epochs 1000 --decay 500 --loss '1*MSE' --save DBPN_BI_X4_jpg_mse --save_results --reset --ext sep-reset 



# +

# 14
python main.py --model DBPN --scale 2 --batch_size 20 --patch_size 64 --epochs 1000 --decay 500 --loss '1*MSE' --save DBPN_BD_X2_jpg_mse --save_results --reset --ext sep-reset 
# -

python main.py --model DBPN --scale 4 --batch_size 20 --patch_size 128 --epochs 1000 --decay 500 --loss '1*MSE' --save DBPN_BD_X4_jpg_mse --save_results --reset --ext sep-reset


# 10008번 cp2
python main.py --model DBPN --scale 2 --batch_size 20 --patch_size 64 --epochs 1000 --decay 500 --loss '1*MSE' --save DBPN_DN_X2_jpg_mse --save_results --reset --ext sep-reset 


#10007번 cp2
python main.py --model DBPN --scale 4 --batch_size 20 --patch_size 128 --epochs 1000 --decay 500 --loss '1*MSE' --save DBPN_DN_X4_jpg_mse --save_results --reset --ext sep-reset
