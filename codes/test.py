# +
import argparse
import os

import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random
from glob import glob
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# -

parser = argparse.ArgumentParser(description='SR+OD implementation')
parser.add_argument("--SR", type=str, default='EDSR', 
                    help="EDSR, DRRN, RCAN, DBPN, MSRN, ESRGAN")
parser.add_argument("--OD", type=str, default='FasterRCNN', 
                    help="detectron2 models")
parser.add_argument("--degradation", type=str, default='BI',
                   help="BI, BD, DN")
parser.add_argument("--scale", type=int, default=2)

args = parser.parse_args()

if args.SR=="EDSR":
    cmd = "python edsr/main.py --model EDSR --scale {} \
    --n_resblocks 32 --n_feats 256 --res_scale 0.1 \
    --dir_demo input_sample/ \
    --test_only \
    --save_results \
    --pre_train edsr/weights/EDSR_{}_X{}.pt".format(
        args.scale, args.degradation, args.scale)
    os.system(cmd)

elif args.SR=="DRRN":
    cmd = "python edsr/main.py --template DRRN --scale {} \
    --dir_demo '../input_sample' \
    --test_only \
    --save_results \
    --pre_train 'weights/DRRN_{}.pt".format(args.scale, args.degradation)
    os.system(cmd)

elif args.SR=="RCAN":
    patch = 96 if args.scale==2 else 192
    cmd = "python edsr/main.py --model RCAN --scale {} \
    --patch_size {} --n_resgroups 10 --n_resblocks 20 \
    --dir_demo '../input_sample' \
    --test_only \
    --save_results \
    --pre_train 'weights/RCAN_{}_X{}.pt".format(
        args.scale, patch, args.degradation, args.scale)
    os.system(cmd)

elif args.SR=="DBPN":
    patch = 64 if args.scale==2 else 128
    cmd = "python edsr/main.py --model DBPN --scale {} \
    --patch_size {} \
    --dir_demo '../input_sample' \
    --test_only \
    --save_results \
    --pre_train 'weights/DBPN_{}_X{}.pt".format(
        args.scale, patch, args.degradation, args.scale)
    os.system(cmd)

elif args.SR=="MSRN":
    patch = 128 if args.scale==2 else 256
    cmd = "python edsr/main.py --model MSRN --scale {} \
    --patch_size {} \
    --dir_demo '../input_sample' \
    --test_only \
    --save_results \
    --pre_train 'weights/MSRN_{}_X{}.pt".format(
        args.scale, patch, args.degradation, args.scale)
    os.system(cmd)

elif args.SR=="ESRGAN" or args.SR=="RRDB":
    arch_file = "RRDBNet_arch_x2.py" if args.scale==2 else "RRDBNet_arch_x4.py"
    cp = "\cp esrgan/models/archs/{} esrgan/models/archs/RRDBNet_arch.py".format(arch_file)
    os.system(cp)
    cmd = "python esrgan/test.py \
    -opt esrgan/options/test/{}_{}_X{}.yml".format(
        args.SR, args.degradation, args.scale
    )
    os.system(cmd)

else:
    raise NotImplementedError

    
