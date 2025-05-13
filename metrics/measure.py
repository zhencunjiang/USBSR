from collections import OrderedDict

import numpy as np

import lpips as lp

from .psnr import psnr
from .ssim import calculate_ssim as ssim
from .best_psnr import best_psnr
import pyiqa


class IQA:
    referecnce_metrics = ["psnr", "ssim", "best_psnr", "best_ssim", "lpips"]
    nonreference_metrics = ["niqe", "pi", "brisque", 'musiq']
    supported_metrics = referecnce_metrics + nonreference_metrics

    def __init__(self, metrics, lpips_type="alex", cuda=True):
        for metric in self.supported_metrics:
            if not (metric in self.supported_metrics):
                raise KeyError(
                    "{} is not Supported metric. (Support only {})".format(
                        metric, self.supported_metrics
                    )
                )

        if "lpips" in metrics:
            self.lpips_fn = lp.LPIPS(net=lpips_type)
            self.cuda = cuda
            if cuda:
                self.lpips_fn = self.lpips_fn.cuda()


        self.niqe_metric = pyiqa.create_metric('niqe').cuda()
        self.brisque_metric = pyiqa.create_metric('brisque').cuda()
        self.musiq_metric = pyiqa.create_metric('musiq').cuda()
        self.pi_metric = pyiqa.create_metric('pi').cuda()


    def __call__(self, res, ref=None, metrics=["niqe"]):
        """
        res, ref: [0, 255]
        """

        scores = OrderedDict()
        for metric in metrics:
            if metric in self.referecnce_metrics:
                if ref is None:
                    raise ValueError(
                        "Ground-truth refernce is needed for {}".format(metric)
                    )
                scores[metric] = getattr(self, "calculate_{}".format(metric))(res, ref)

            elif metric in self.nonreference_metrics:
                scores[metric] = getattr(self, "calculate_{}".format(metric))(res)

            else:
                raise KeyError(
                    "{} is not Supported metric. (Support only {})".format(
                        metric, self.supported_metrics
                    )
                )
        return scores

    def calculate_lpips(self, res, ref):
        if res.ndim < 3:
            return 0
        res = lp.im2tensor(res)
        ref = lp.im2tensor(ref)
        if self.cuda:
            res = res.cuda()
            ref = ref.cuda()
        score = self.lpips_fn(res, ref)
        return score.item()

    def calculate_niqe(self, res):

        return self.niqe_metric(res)

    def calculate_brisque(self, res):

        return self.brisque_metric(res)

    def calculate_musiq(self, res):

        return self.musiq_metric(res)

    def calculate_pi(self, res):

        return self.pi_metric(res)

    def calculate_best_psnr(self, res, ref):
        best_psnr_, best_ssim_ = best_psnr(res, ref)
        self.best_ssim = best_ssim_
        return best_psnr_

    def calculate_best_ssim(self, res, ref):
        assert hasattr(self, "best_ssim")
        return self.best_ssim

    @staticmethod
    def calculate_psnr(res, ref):
        return psnr(res, ref)

    @staticmethod
    def calculate_ssim(res, ref):
        return ssim(res, ref)
