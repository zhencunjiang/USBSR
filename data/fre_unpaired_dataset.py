import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import utils as util
from utils.registry import DATASET_REGISTRY


import pywt
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import torch
import torch.nn as  nn

from numpy import linalg as la


def svd_denoise(img):
    img=img[:,:,0]
    u, sigma, vt = la.svd(img)
    h, w = img.shape[:2]
    h1 = int(h * 0.1) #取前10%的奇异值重构图像
    sigma1 = np.diag(sigma[:h1],0) #用奇异值生成对角矩阵
    u1 = np.zeros((h,h1), float)
    u1[:,:] = u[:,:h1]
    vt1 = np.zeros((h1,w), float)
    vt1[:,:] = vt[:h1,:]
    return u1.dot(sigma1).dot(vt1)





def decompose_image(image):
    image=image[:,:,0]
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    low_freq = pywt.idwt2((LL, (np.zeros_like(LH), np.zeros_like(HL), np.zeros_like(HH))), 'haar')
    high_freq = pywt.idwt2((np.zeros_like(LL), (LH, HL, HH)), 'haar')
    return high_freq,low_freq
def ret(image):
    height, width=image.shape[0],image.shape[1]
    rgb_image = np.zeros((height, width, 3))
    rgb_image[:, :, 0] = image
    rgb_image[:, :, 1] = image
    rgb_image[:, :, 2] = image
    return rgb_image

@DATASET_REGISTRY.register()
class FreUnPairedDataset(data.Dataset):
    """
    Read unpaired reference images, i.e., source (src) and target (tgt),
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.src_paths, self.src_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_src"]
        )
        self.tgt_paths, self.tgt_sizes = util.get_image_paths(
            opt["data_type"], opt["dataroot_tgt"]
        )

        if opt.get("ratios"):
            ratio_src, ratio_tgt = opt["ratios"]
            self.src_paths *= ratio_src; self.src_sizes *= ratio_src
            self.tgt_paths *= ratio_tgt; self.tgt_sizes *= ratio_tgt

        merged_src = list(zip(self.src_paths, self.src_sizes))
        random.shuffle(merged_src)
        self.src_paths[:], self.src_sizes[:] = zip(*merged_src)

        if opt["data_type"] == "lmdb":
            self.lmdb_envs = False

    def _init_lmdb(self, dataroots):
        envs = []
        for dataroot in dataroots:
            envs.append(
                lmdb.open(
                    dataroot, readonly=True, lock=False, readahead=False, meminit=False
                )
            )
        self.lmdb_envs = True
        return envs

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb" and (not self.lmdb_envs):
            self.src_env, self.tgt_env = self._init_lmdb(
                [
                    self.opt["dataroot_src"],
                    self.opt["dataroot_tgt"],
                ]
            )

        scale = self.opt["scale"]
        cropped_src_size, cropped_tgt_size = self.opt["src_size"], self.opt["tgt_size"]

        # get tgt image
        tgt_path = self.tgt_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.tgt_sizes[index].split("_")]
        else:
            resolution = None
        img_tgt = util.read_img(
            self.tgt_env, tgt_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_tgt = util.modcrop(img_tgt, scale)

        # get src image
        src_path = self.src_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.src_sizes[index].split("_")]
        else:
            resolution = None
        img_src = util.read_img(self.src_env, src_path, resolution)
        # im_haar = denoise_wavelet(img_src, wavelet='db3', channel_axis=0)
        # noisy_harr = img_src - im_haar
        # im_svd = svd_denoise(img_src)
        # noisy_svd = img_src - np.expand_dims(im_svd, axis=2)



        if self.opt["phase"] == "train":
            # assert (
            #     cropped_src_size == cropped_tgt_size // scale
            # ), "tgt size does not match src size"

            # randomly crop
            H, W, C = img_src.shape
            rnd_h = random.randint(0, max(0, H - cropped_src_size))
            rnd_w = random.randint(0, max(0, W - cropped_src_size))
            img_src = img_src[
                rnd_h : rnd_h + cropped_src_size, rnd_w : rnd_w + cropped_src_size
            ]


            H, W, C = img_tgt.shape
            rnd_h = random.randint(0, max(0, H - cropped_tgt_size))
            rnd_w = random.randint(0, max(0, W - cropped_tgt_size))
            img_tgt = img_tgt[
                rnd_h : rnd_h + cropped_tgt_size, rnd_w : rnd_w + cropped_tgt_size
            ]

            # augmentation - flip, rotate
            img_tgt = util.augment(
                [img_tgt],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

            img_src = util.augment(
                [img_src],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )




        # change color space if necessary
        if self.opt["color"]:
            # TODO during val no definition
            img_src, img_tgt = util.channel_convert(self.opt["color"], [img_src, img_tgt])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_src.shape[2] == 3:
            img_src = img_src[:, :, [2, 1, 0]]
            img_tgt = img_tgt[:, :, [2, 1, 0]]



        hf_tgt, lf_tgt = decompose_image(img_tgt)
        hf_src, lf_src = decompose_image(img_src)
        hf_tgt=ret(hf_tgt)
        lf_tgt=ret(lf_tgt)
        hf_src=ret(hf_src)
        lf_src=ret(lf_src)
        img_src = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_src, (2, 0, 1)))
        ).float()
        img_tgt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_tgt, (2, 0, 1)))
        ).float()
        img_hf_tgt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(hf_tgt, (2, 0, 1)))
        ).float()
        img_lf_tgt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(lf_tgt, (2, 0, 1)))
        ).float()
        img_hf_src = torch.from_numpy(
            np.ascontiguousarray(np.transpose(hf_src, (2, 0, 1)))
        ).float()
        img_lf_src = torch.from_numpy(
            np.ascontiguousarray(np.transpose(lf_src, (2, 0, 1)))
        ).float()
        # noise_wave=torch.from_numpy(
        #     np.ascontiguousarray(np.transpose(noisy_harr, (2, 0, 1)))
        # ).float()
        # noise_svd=torch.from_numpy(
        #     np.ascontiguousarray(np.transpose(noisy_svd, (2, 0, 1)))
        # ).float()



        data_dict = {
            "src": img_src,
            "tgt": img_tgt,
            "hf_tgt": img_hf_tgt,
            "lf_tgt": img_lf_tgt,
            "hf_src": img_hf_src,
            "lf_src": img_lf_src,
            "src_path": src_path,
            "tgt_path": tgt_path
            # 'noise_wave':noise_wave,
            # 'noise_svd': noise_svd,

        }
        return data_dict

    def __len__(self):
        return len(self.src_paths)
