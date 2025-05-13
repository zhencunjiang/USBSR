import logging
from collections import OrderedDict
import random

import torch
import torch.nn as nn
from kornia.color import rgb_to_grayscale
import torch.nn.functional as F
from utils.registry import MODEL_REGISTRY
import numpy as np
from .base_model import BaseModel
import pywt
import cv2
from skimage.util import random_noise

logger = logging.getLogger("base")

import torch
import torch.nn.functional as F
import random


def ret_tr(image):
    height, width=image.shape[0],image.shape[1]
    rgb_image = torch.zeros((3,height, width))
    rgb_image[0,:, :] = image
    rgb_image[1,:, :] = image
    rgb_image[2,:, :] = image
    return rgb_image
def gaussian_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1, size // 2 + 1)
    x = x.float().view(-1, 1)
    y = x.t()
    kernel_2d = torch.exp(-0.5 * (x ** 2 + y ** 2) / sigma ** 2)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d


def gaussian_blur(image, blur_kernel_size, blur_sigma):
    # Convert the image to a tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    kernel = gaussian_kernel(blur_kernel_size, blur_sigma)
    kernel = kernel.view(1, 1, blur_kernel_size, blur_kernel_size)
    kernel = kernel.repeat(image.shape[1], 1, 1, 1).cuda()

    image = F.pad(image, (blur_kernel_size // 2, blur_kernel_size // 2, blur_kernel_size // 2, blur_kernel_size // 2),
                  mode='reflect')
    blurred_image = F.conv2d(image, kernel, groups=image.shape[1]).squeeze(0)
    return blurred_image


def add_speckle_noise(image, sigma):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    # Convert to grayscale by taking the first channel
    grayscale_image = image[0, :, :]
    # Generate speckle noise
    noise = torch.randn(grayscale_image.size()).cuda() * sigma
    # Add speckle noise
    noisy_image = grayscale_image + grayscale_image * noise
    # Clamp the values to be in the range [0, 1]
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image

from torchvision.transforms import Resize
def nvl(image,noise1,noise2):
    noise1=Resize([64,64])(noise1)
    noise2=Resize([64,64])(noise2)
    noise1=noise1[0,:,:]
    noise2 = noise2[0, :, :]
    image= image[0, :, :]
    random_alpha = random.random()
    base_noise = random_alpha* noise1 + (1 - random_alpha) * noise2
    noisy_image=torch.clamp((image+base_noise),0,1)
    noisy_image=ret_tr(noisy_image)
    # print(noisy_image)
    return noisy_image




def deg_image(image):
    out=[]
    for i in range(image.shape[0]):
        blur_k = random.choice([7, 21])
        n_s = random.uniform(0, 0.5)
        blur_s = random.random()
        img=image[i]
        img=gaussian_blur(img, blur_kernel_size=blur_k, blur_sigma=blur_s)
        img = add_speckle_noise(img, sigma=n_s)
        out.append(img)

    return torch.stack(out,dim=0).cuda()


def deg_nvl(image,noise1,noise2):
    out=[]
    for i in range(image.shape[0]):
        blur_k = random.choice([7, 21])
        blur_s = random.random()
        img=image[i]
        img=gaussian_blur(img, blur_kernel_size=blur_k, blur_sigma=blur_s)
        img = nvl(img,noise1[i],noise2[i])
        out.append(img)

    return torch.stack(out,dim=0).cuda()

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = torch.clamp(input, 0, 1)
        output = (output * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
def decompose_image(image):
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
def batch_decompose(image):
    batch_size=image.shape[0]
    hf_all = torch.zeros(batch_size, image.shape[1], image.shape[2],  image.shape[3])
    lf_all = torch.zeros(batch_size, image.shape[1], image.shape[2],  image.shape[3])

    for i in range(batch_size):

        image_np = image[i, 0, :, :].cpu().numpy()
        image_hf,image_lf=decompose_image(image_np)
        image_hf=ret(image_hf)
        image_lf=ret(image_lf)

        hf_all[i, :, :, :] = torch.from_numpy(np.transpose(image_hf, (2, 0, 1))).cuda()
        lf_all[i, :, :, :] = torch.from_numpy(np.transpose(image_lf, (2, 0, 1))).cuda()
    return hf_all,lf_all


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


def expand_tensor(input_tensor):
    # 获取张量的形状
    b, c, h, w = input_tensor.size()
    if c == 1:
        expanded_tensor = input_tensor.repeat(1, 3, 1, 1)
    if c == 3:
        expanded_tensor = input_tensor
    return expanded_tensor


@MODEL_REGISTRY.register()
class FreDegSRModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["src", "tgt"]

        self.network_names = ["netDeg", "netSR", "netD1", "netD2"]
        self.networks = {}

        self.loss_names = [
            'cycle_loss',
            "lr_adv",
            "sr_adv",
            "sr_pix_trans",
            "sr_pix_sr",
            "sr_percep",
            "lr_quant",
            "lr_gauss",
            "noise_mean",
            "color"
        ]
        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        assert set(defined_network_names).issubset(set(self.network_names))

        for name in defined_network_names:
            setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)

        if self.is_train:
            train_opt = opt["train"]
            # setup loss, optimizers, schedulers
            self.setup_train(opt["train"])

            self.max_grad_norm = train_opt["max_grad_norm"]
            self.quant = Quantization()
            self.D_ratio = train_opt["D_ratio"]
            self.optim_sr = train_opt["optim_sr"]
            self.optim_deg = train_opt["optim_deg"]
            self.gray_dis = train_opt["gray_dis"]

            ## buffer
            self.fake_lr_buffer = ShuffleBuffer(train_opt["buffer_size"])
            self.fake_hr_buffer = ShuffleBuffer(train_opt["buffer_size"])

    def feed_data(self, data):

        self.hf_hr = expand_tensor(data["hf_tgt"]).to(self.device)
        self.lf_hr = expand_tensor(data["lf_tgt"]).to(self.device)
        self.hf_lr = expand_tensor(data["hf_src"]).to(self.device)
        self.lf_lr = expand_tensor(data["lf_src"]).to(self.device)
        self.syn_hr = expand_tensor(data["tgt"]).to(self.device)
        self.real_lr = expand_tensor(data["src"]).to(self.device)



        self.real_lr = F.interpolate(self.real_lr, scale_factor=1 / 4, mode="bicubic", align_corners=False)

        # print(self.syn_hr.size(),self.real_lr.shape)

    def deg_forward(self):
        (
            self.fake_real_lr,
            self.predicted_kernel,
            self.predicted_noise,
        ) = self.netDeg(self.hf_hr,self.lf_hr)
        if self.losses.get("sr_pix_trans"):
            self.fake_real_lr_quant = self.quant(self.fake_real_lr)
            self.syn_sr = self.netSR(self.fake_real_lr_quant)

    def sr_forward(self):
        if not self.optim_deg:
            (
                self.fake_real_lr,
                self.predicted_kernel,
                self.predicted_noise,
            ) = self.netDeg(self.syn_hr)


        self.fake_real_lr = deg_image(self.fake_real_lr)
        self.fake_real_lr_quant = self.quant(self.fake_real_lr)
        self.syn_sr = self.netSR(self.fake_real_lr_quant.detach())

    def cycle_forward(self):
        self.lr_sr=self.netSR(self.quant(self.real_lr)).detach()
        self.lr_sr_hf,self.lr_sr_lf=batch_decompose(self.lr_sr)
        self.real_lr_deg, _, _ = self.netDeg(self.lr_sr_hf,self.lr_sr_lf)

    def optimize_trans_models(self, step, loss_dict):

        self.set_requires_grad(["netDeg"], True)
        self.deg_forward()
        if self.losses.get("cycle_loss"):
           self.cycle_forward()
        loss_G = 0

        if self.losses.get("lr_adv"):
            self.set_requires_grad(["netD1"], False)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            g1_adv_loss = self.calculate_gan_loss_G(
                self.netD1, self.losses["lr_adv"], real, fake
            )
            loss_dict["g1_adv"] = g1_adv_loss.item()
            loss_G += self.loss_weights["lr_adv"] * g1_adv_loss

        if self.losses.get("cycle_loss"):
            self.set_requires_grad(["netD1"], False)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            redeg_adv_loss = self.calculate_gan_loss_G(
                self.netD1, self.losses["lr_adv"], real, self.real_lr_deg
            )
            loss_G += self.loss_weights["lr_adv"] * (redeg_adv_loss)

        if self.losses.get("sr_pix_trans"):
            self.set_requires_grad(["netSR"], False)
            sr_pix = self.losses["sr_pix_trans"](self.syn_hr, self.syn_sr)
            loss_dict["sr_pix_trans"] = sr_pix.item()
            loss_G += self.loss_weights["sr_pix_trans"] * sr_pix

        if self.losses.get("cycle_loss"):
            c_loss = self.losses["cycle_loss"](self.real_lr, self.real_lr_deg)
            loss_dict["cycle_loss"] = c_loss.item()
            loss_G += c_loss * self.loss_weights["cycle_loss"]

        if self.losses.get("noise_mean"):
            noise = self.predicted_noise
            noise_mean = (
                self.losses["noise_mean"](noise, torch.zeros_like(noise))
            )
            loss_dict["noise_mean"] = noise_mean.item()
            loss_G += self.loss_weights["noise_mean"] * noise_mean

        self.set_optimizer(names=["netDeg"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["netDeg"], self.max_grad_norm)
        self.set_optimizer(names=["netDeg"], operation="step")

        ## update D
        if step % self.D_ratio == 0:
            self.set_requires_grad(["netD1"], True)
            if self.gray_dis:
                real = rgb_to_grayscale(self.real_lr)
                fake = rgb_to_grayscale(self.fake_real_lr)
            else:
                real = self.real_lr
                fake = self.fake_real_lr
            loss_d1 = self.calculate_gan_loss_D(
                self.netD1, self.losses["lr_adv"],
                real, self.fake_lr_buffer.choose(fake)
            )
            loss_dict["d1_adv"] = loss_d1.item()
            loss_D = self.loss_weights["lr_adv"] * loss_d1
            self.optimizers["netD1"].zero_grad()
            loss_D.backward()
            self.clip_grad_norm(["netD1"], self.max_grad_norm)
            self.optimizers["netD1"].step()

        return loss_dict

    def optimize_sr_models(self, step, loss_dict):

        self.set_requires_grad(["netSR"], True)
        self.set_requires_grad(["netDeg"], False)
        self.sr_forward()
        loss_G = 0

        if self.losses.get("sr_adv"):
            self.set_requires_grad(["netD2"], False)
            sr_adv_loss = self.calculate_gan_loss_G(
                self.netD2, self.losses["sr_adv"],
                self.syn_hr, self.syn_sr
            )
            loss_dict["sr_adv"] = sr_adv_loss.item()
            loss_G += self.loss_weights["sr_adv"] * sr_adv_loss

        if self.losses.get("sr_percep"):
            sr_percep, sr_style = self.losses["sr_percep"](
                self.syn_hr, self.syn_sr
            )
            loss_dict["sr_percep"] = sr_percep.item()
            if sr_style is not None:
                loss_dict["sr_style"] = sr_style.item()
                loss_G += self.loss_weights["sr_percep"] * sr_style
            loss_G += self.loss_weights["sr_percep"] * sr_percep

        if self.losses.get("sr_pix_sr"):
            sr_pix = self.losses["sr_pix_sr"](self.syn_hr, self.syn_sr)
            loss_dict["sr_pix_sr"] = sr_pix.item()
            loss_G += self.loss_weights["sr_pix_sr"] * sr_pix

        self.set_optimizer(names=["netSR"], operation="zero_grad")
        loss_G.backward()
        self.clip_grad_norm(["netSR"], self.max_grad_norm)
        self.set_optimizer(names=["netSR"], operation="step")

        ## update D2
        if step % self.D_ratio == 0:
            if self.losses.get("sr_adv"):
                self.set_requires_grad(["netD2"], True)
                loss_d2 = self.calculate_gan_loss_D(
                    self.netD2, self.losses["sr_adv"],
                    self.syn_hr, self.fake_hr_buffer.choose(self.syn_sr)
                )
                loss_dict["d2_adv"] = loss_d2.item()
                loss_D = self.loss_weights["sr_adv"] * loss_d2
                self.optimizers["netD2"].zero_grad()
                loss_D.backward()
                self.clip_grad_norm(["netD2"], self.max_grad_norm)
                self.optimizers["netD2"].step()

        return loss_dict

    def optimize_parameters(self, step):
        loss_dict = OrderedDict()

        # optimize trans
        if self.optim_deg:
            loss_dict = self.optimize_trans_models(step, loss_dict)

        # optimize SR
        if self.optim_sr:
            loss_dict = self.optimize_sr_models(step, loss_dict)

        self.log_dict = loss_dict

    def calculate_gan_loss_D(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake.detach())
        d_pred_real = netD(real)

        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)

        return (loss_real + loss_fake) / 2

    def calculate_gan_loss_G(self, netD, criterion, real, fake):

        d_pred_fake = netD(fake)
        loss_real = criterion(d_pred_fake, True, is_disc=False)

        return loss_real

    def test(self, test_data, crop_size=None):
        self.src = expand_tensor(test_data["src"]).to(self.device)
        if test_data.get("tgt") is not None:
            self.tgt = expand_tensor(test_data["tgt"]).to(self.device)

        self.set_network_state(["netSR"], "eval")
        with torch.no_grad():
            if crop_size is None:
                self.fake_tgt = self.netSR(self.src)
            else:
                self.fake_tgt = self.crop_test(self.src, crop_size)
        self.set_network_state(["netSR"], "train")

        if hasattr(self, "netDeg"):
            self.set_network_state(["netDeg"], "eval")
            if hasattr(self, "tgt"):
                with torch.no_grad():
                    self.fake_lr = self.netDeg(self.tgt)[0]
            self.set_network_state(["netDeg"], "train")

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["lr"] = self.src.detach()[0].float().cpu()
        out_dict["sr"] = self.fake_tgt.detach()[0].float().cpu()
        if hasattr(self, "fake_lr"):
            out_dict["fake_lr"] = self.fake_lr.detach()[0].float().cpu()
        return out_dict

    def crop_test(self, lr, crop_size):
        b, c, h, w = lr.shape
        scale = self.opt["scale"]

        h_start = list(range(0, h - crop_size, crop_size))
        w_start = list(range(0, w - crop_size, crop_size))

        sr1 = torch.zeros(b, c, int(h * scale), int(w * scale), device=self.device) - 1
        for hs in h_start:
            for ws in w_start:
                lr_patch = lr[:, :, hs: hs + crop_size, ws: ws + crop_size]
                sr_patch = self.netSR(lr_patch)

                sr1[:, :,
                int(hs * scale):int((hs + crop_size) * scale),
                int(ws * scale):int((ws + crop_size) * scale)
                ] = sr_patch

        h_end = list(range(h, crop_size, -crop_size))
        w_end = list(range(w, crop_size, -crop_size))

        sr2 = torch.zeros(b, c, int(h * scale), int(w * scale), device=self.device) - 1
        for hd in h_end:
            for wd in w_end:
                lr_patch = lr[:, :, hd - crop_size:hd, wd - crop_size:wd]
                sr_patch = self.netSR(lr_patch)

                sr2[:, :,
                int((hd - crop_size) * scale):int(hd * scale),
                int((wd - crop_size) * scale):int(wd * scale)
                ] = sr_patch

        mask1 = (
                (sr1 == -1).float() * 0 +
                (sr2 == -1).float() * 1 +
                ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        mask2 = (
                (sr1 == -1).float() * 1 +
                (sr2 == -1).float() * 0 +
                ((sr1 > 0) * (sr2 > 0)).float() * 0.5
        )

        sr = mask1 * sr1 + mask2 * sr2

        return sr


class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.
    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.
        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.
        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        if self.buffer_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images