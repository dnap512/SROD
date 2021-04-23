import argparse

# Some basic setup:
# Setup detectron2 logger
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


def parse_args(args) -> argparse:
    parser = argparse.ArgumentParser(description='SR+OD implementation using Detectron2')

    parser.add_argument('--input-dir', type=str, help='directory path', default="./output_sr")

    parser.add_argument('--sr-save', type=str, help='save sr result directory path', default="./output_sr")
    parser.add_argument('--od-save', type=str, help='save od result directory path', default="./output_od")

    parser.add_argument("--SR", type=str, default='EDSR', help="EDSR, DRRN, RCAN, DBPN, MSRN, ESRGAN")
    parser.add_argument("--OD", type=str, default='fasterRCNN', help="fasterRCNN, retinanet, maskRCNN")

    parser.add_argument("--degradation", type=str, default='BI', help="BI, BD, DN")
    parser.add_argument("--scale", type=int, default=2)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def sr_perform(sr_model: str, degradation: str, scale: int) -> None:
    if sr_model == "EDSR":
        cmd = "python edsr/main.py --model EDSR --scale {} \
        --n_resblocks 32 --n_feats 256 --res_scale 0.1 \
        --dir_demo input_sample/ \
        --test_only \
        --save_results \
        --pre_train edsr/weights/EDSR_{}_X{}.pt".format(scale, degradation, scale)
        os.system(cmd)

    elif sr_model == "DRRN":
        cmd = "python edsr/main.py --template DRRN --scale {} \
        --dir_demo '../input_sample' \
        --test_only \
        --save_results \
        --pre_train 'weights/DRRN_{}.pt".format(scale, degradation)
        os.system(cmd)

    elif sr_model == "RCAN":
        patch = 96 if scale == 2 else 192
        cmd = "python edsr/main.py --model RCAN --scale {} \
        --patch_size {} --n_resgroups 10 --n_resblocks 20 \
        --dir_demo '../input_sample' \
        --test_only \
        --save_results \
        --pre_train 'weights/RCAN_{}_X{}.pt".format(scale, patch, degradation, scale)
        os.system(cmd)

    elif sr_model == "DBPN":
        patch = 64 if scale == 2 else 128
        cmd = "python edsr/main.py --model DBPN --scale {} \
        --patch_size {} \
        --dir_demo '../input_sample' \
        --test_only \
        --save_results \
        --pre_train 'weights/DBPN_{}_X{}.pt".format(scale, patch, degradation, scale)
        os.system(cmd)

    elif sr_model == "MSRN":
        patch = 128 if scale == 2 else 256
        cmd = "python edsr/main.py --model MSRN --scale {} \
        --patch_size {} \
        --dir_demo '../input_sample' \
        --test_only \
        --save_results \
        --pre_train 'weights/MSRN_{}_X{}.pt".format(scale, patch, degradation, scale)
        os.system(cmd)

    elif sr_model == "ESRGAN" or sr_model == "RRDB":
        arch_file = "RRDBNet_arch_x2.py" if scale == 2 else "RRDBNet_arch_x4.py"
        cp = "\cp esrgan/models/archs/{} esrgan/models/archs/RRDBNet_arch.py".format(arch_file)
        os.system(cp)
        cmd = "python esrgan/test.py \
        -opt esrgan/options/test/{}_{}_X{}.yml".format(sr_model, degradation, scale)
        os.system(cmd)

    else:
        raise NotImplementedError



def config(od_model: str) -> DefaultPredictor:

    # model config
    if od_model == "fasterRCNN":
        od_model = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

    elif od_model == "retinanet":
        od_model = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

    elif od_model == "maskRCNN":
        od_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    else:
        raise NotImplementedError

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(od_model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(od_model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg), cfg


def inference(image_path: str, save_path: str, predictor: DefaultPredictor, cfg: get_cfg) -> None:
    images = glob(image_path + '/*')

    for image in images:
        read_img = cv2.imread(image)
        img_name = image.split("/")[-1]

        # inference
        outputs = predictor(read_img)

        # Visualization
        v = Visualizer(read_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # save image
        cv2.imwrite(save_path + "/" + img_name, out.get_image()[:, :, ::-1])


def main(args=None):
    setup_logger()
    args = parse_args(args)

    # variable definition
    od_model = args.OD
    sr_model = args.SR

    degradation = args.degradation
    scale = args.scale

    sr_save = args.sr_save
    od_save = args.od_save

    Path(sr_save).mkdir(parents=True, exist_ok=True)
    Path(od_save).mkdir(parents=True, exist_ok=True)

    # super resolution start
    sr_perform(sr_model, degradation, scale)

    # object detection start
    predictor, cfg = config(od_model)
    inference(sr_save, od_save, predictor, cfg)


if __name__ == '__main__':
    main()
