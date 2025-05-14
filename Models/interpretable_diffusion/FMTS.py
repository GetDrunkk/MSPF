import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from Models.interpretable_diffusion.transformer import Transformer
import os
from typing import Optional


class FM_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            **kwargs
    ):
        super(FM_TS, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)


        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))
    
    def output(self, x, t, mask:Optional[torch.Tensor]=None, padding_masks=None):
        """
        x   : [B, D, T]     (缺口处已置 0)
        mask: [B, T]  True=观测 False=缺口   (C1 训练/推理时提供，B0 为 None)
        """
        if mask is not None:
            # 只有当 x 里还没包含 mask 通道时才拼接
            if x.shape[1] == self.feature_size - 1:     # 自适应判断
                x = torch.cat([x, mask.unsqueeze(1).float()], dim=1)
        return self.model(x, t, padding_masks=None)



    @torch.no_grad()
    def sample(self, shape):
        assert self.feature_size == 7, "Unconditional sampling (sample / generate_mts) 仅在 B0 使用"
        self.eval()
        zt = torch.randn(shape, device=self.device)
        timesteps = torch.linspace(0, 1, self.num_timesteps+1, device=self.device)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(zt.clone(),
                            torch.full((shape[0],), t_curr*self.time_scalar,
                                    device=self.device),
                            mask=None)
            zt = zt + step * v
        return zt




    def generate_mts(self, batch_size=16):
        feature_size, seq_length = self.feature_size, self.seq_length
        return self.sample((batch_size, seq_length, feature_size))




    def _train_loss(self, x_start, mask=None):
        """
        mask: Bool [B,T]  True=观测  False=缺口
              若为 None (B0) 则在所有位置计算损失
        """
        z0 = torch.randn_like(x_start)
        z1 = x_start
        t  = torch.rand(z0.shape[0],1,1, device=z0.device)

        z_t    = t * z1 + (1.-t) * z0
        target = z1 - z0

        model_out = self.output(z_t, t.squeeze()*self.time_scalar, mask)

        mse = (model_out - target) ** 2          # [B,D,T]
        if mask is not None:                     # 仅缺口位置
            mse = mse.permute(0,2,1)[~mask]
        else:                                    # B0：全部位置
            mse = mse.view(-1)
        return mse.mean()


    def forward(self, x, mask=None):
        """
        x    : [B, D, T]  (缺口已置 0)
        mask : Bool [B,T]  or None
        """
        return self._train_loss(x_start=x, mask=mask)



    def fast_sample_infill(self, shape, target, partial_mask=None):

        z0 = torch.randn(shape).to(self.device)
        z1 = zt = z0
        for t in range(self.num_timesteps):
            t = t/self.num_timesteps  ## scale to 0-1
            t = t**(float(os.environ['hucfg_Kscale']))  ## perform t-power sampling

            
            z0 = torch.randn(shape).to(self.device)  ## re init the z0

            target_t = target*t + z0*(1-t)  ## get the noisy target
            zt = z1*t + z0*(1-t)  ##
            # import ipdb; ipdb.set_trace()
            mask_1d = partial_mask.any(-1) if partial_mask is not None else None  # [B,T] or None
            v = self.output(zt, torch.tensor([t*self.time_scalar]).to(self.device), mask_1d)


            z1 = zt.clone() + (1 - t) * v  ## one step euler
            z1 = torch.clamp(z1, min=-1, max=1) ## make sure the upper and lower bound dont exceed


        return z1






