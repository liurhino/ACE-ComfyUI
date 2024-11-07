# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from dataclasses import dataclass, field
from scepter.modules.model.registry import NOISE_SCHEDULERS
from scepter.modules.model.diffusion.schedules import BaseNoiseScheduler
from scepter.modules.model.diffusion.util import _i

@dataclass
class ScheduleOutput(object):
    x_t: torch.Tensor
    x_0: torch.Tensor
    t: torch.Tensor
    sigma: torch.Tensor
    alpha: torch.Tensor
    custom_fields: dict = field(default_factory=dict)

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


@NOISE_SCHEDULERS.register_class()
class LinearScheduler(BaseNoiseScheduler):
    para_dict = {}

    def init_params(self):
        super().init_params()
        self.beta_min = self.cfg.get('BETA_MIN', 0.00085)
        self.beta_max = self.cfg.get('BETA_MAX', 0.012)

    def betas_to_sigmas(self, betas):
        return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))

    def get_schedule(self):
        betas = torch.linspace(self.beta_min,
                               self.beta_max,
                               self.num_timesteps,
                               dtype=torch.float32)
        sigmas = self.betas_to_sigmas(betas)
        self._sigmas = sigmas
        self._betas = betas
        self._alphas = torch.sqrt(1 - sigmas**2)
        self._timesteps = torch.arange(len(sigmas), dtype=torch.float32)

    def add_noise(self, x_0, noise=None, t=None, **kwargs):
        if t is None:
            t = torch.randint(0,
                              self.num_timesteps, (x_0.shape[0], ),
                              device=x_0.device).long()
        alpha = _i(self.alphas, t, x_0)
        sigma = _i(self.sigmas, t, x_0)
        x_t = alpha * x_0 + sigma * noise

        return ScheduleOutput(x_0=x_0, x_t=x_t, t=t, alpha=alpha, sigma=sigma)