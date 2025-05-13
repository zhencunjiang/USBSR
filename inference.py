import argparse
import logging
import math
import os
import os.path as osp
import random
import sys
import cv2
from collections import defaultdict
from glob import glob
from tqdm import tqdm
import pyiqa
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

sys.path.append("../../")
import utils as util
import utils.option as option
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from metrics import IQA
from models import create_model

def gaussian_blur(image, blur_kernel_size, blur_sigma):
    return cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size),sigmaX=blur_sigma,sigmaY=blur_sigma)

#### options
parser = argparse.ArgumentParser()
parser.add_argument(
    "-opt",
    type=str,
    default="/home/ps/zhencunjiang/US_sr/USBSR/options/inference.yml",
    help="Path to options YMAL file.",
)
parser.add_argument("-input_dir", type=str, default="/home/ps/zhencunjiang/US_sr/inference_dataset/clinical_lr_ultrasound")
parser.add_argument("-output_dir", type=str, default="/home/ps/zhencunjiang/US_sr/inference_dataset/clinical_lr_ultrasound_infer/ours")
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

model = create_model(opt)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)
niqe_metric = pyiqa.create_metric('niqe').cuda()
musiq_metric = pyiqa.create_metric('musiq').cuda()
pi_metric = pyiqa.create_metric('pi').cuda()
niqe=[]
musiq=[]
pi=[]
test_files = glob(osp.join(args.input_dir, "*"))
for inx, path in tqdm(enumerate(test_files)):
    name = path.split("/")[-1].split(".")[0]

    img = cv2.imread(path)[:, :, [2, 1, 0]]
    img = gaussian_blur(img, 21, 0.9)
    img = img.transpose(2, 0, 1)[None] / 255
    img_t = torch.as_tensor(np.ascontiguousarray(img)).float()

    model.test({"src": img_t})
    outdict = model.get_current_visuals()

    sr = outdict["sr"]
    # print(sr.shape)
    sr_t=sr.unsqueeze(0)
    niqe.append(niqe_metric(sr_t))
    pi.append(pi_metric(sr_t))
    musiq.append(musiq_metric(sr_t))

    sr_im = util.tensor2img(sr)

    save_path = osp.join(args.output_dir, "{}_x{}.png".format(name, opt["scale"]))
    cv2.imwrite(save_path, sr_im)
print("niqe:",sum(niqe)/len(niqe))
print("pi:",sum(pi)/len(niqe))
print("musiq:",sum(musiq)/len(niqe))
