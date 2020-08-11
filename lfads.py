"""LFADS architecture, loss function and Dataset class."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.functional as F
import torch.optim as optim
import numpy as np


class LFADS(nn.Module):
    def __init__(self, hypers, device):
        super(LFADS, self).__init__()
        # network architecture
        self.in_drop = nn.Dropout(p=hypers['drop_rate'])
        self.enc, self.enc_h0 = self.init_enc(hypers['enc'])
        self.ic_w = nn.Linear(2 * hypers['enc']['n_hidden'], 2 * hypers['dec']['n_hidden'] * hypers['dec']['n_layers'])
        self.enc_drop = nn.Dropout(p=hypers['drop_rate'])
        self.dec, self.split_dec_ic = self.init_dec(hypers['dec'])
        self.dec_drop = nn.Dropout(p=hypers['drop_rate'])
        self.factors = nn.utils.weight_norm(nn.Linear(hypers['dec']['n_hidden'], hypers['dec']['n_fact']))
        self.out_w = nn.Linear(hypers['dec']['n_fact'], hypers['enc']['in_features'])

        # for some reason a GRU with 0 inputs cannot be initialized on GPUs,
        # hence a 0 vector will be given as network input at every time point
        self.in_0 = torch.zeros(hypers['seq_len'], hypers['batch_size'], 1).to(device)

    def init_enc(self, hype):
        # *2 bcs it's bidirectional; 1 is the expendable batch dim
        h0 = nn.Parameter(torch.randn(2 * hype['n_layers'], 1, hype['n_hidden'], dtype=torch.float)
                          / np.sqrt(hype['n_hidden']))
                gru = nn.GRU(hype['in_features'], hype['n_hidden'], hype['n_layers'],
                     bidirectional=True)
        return gru, h0

    def init_dec(self, hype):
        def split_dec_ic(g0):
            g0 = g0.unsqueeze(0)
            return torch.cat(torch.chunk(g0, chunks=hype['n_layers'], dim=-1))

        dec = nn.GRU(1, hype['n_hidden'], hype['n_layers'], bidirectional=False)
        return dec, split_dec_ic

    def encode(self, x):
        x = self.in_drop(x)
        h0 = self.enc_h0.repeat(1, x.shape[1], 1)
        e_all, e = self.enc(x, h0)
        e_all = self.enc_drop(e_all)
        ic_mean, ic_logvar = self.ic_w(e_all[-1, :, :]).chunk(2, -1)
        return ic_mean, ic_logvar

    def reparametrize(self, mean, logvar, varmin=1e-16):
        logvar_vm = torch.log(torch.exp(logvar) + varmin)
        return mean + torch.exp(0.5 * logvar_vm) * torch.randn_like(logvar)  # .to(device)

    def decode(self, g0, seq_len):
        g0 = self.split_dec_ic(g0)
        g_all, _ = self.dec(self.in_0, g0)
        y = self.factors(g_all)
        y = self.out_w(y)
        return y

    def forward(self, x):
        ic_mean, ic_logvar = self.encode(x)
        g0 = self.reparametrize(ic_mean, ic_logvar)
        x_rec = self.decode(g0, x.shape[0])
        return x_rec, ic_mean, ic_logvar
    
    def reco_return_factors(self, x):
        ic_mean, ic_logvar = self.encode(x)
        g0 = self.reparametrize(ic_mean, ic_logvar)
        g0 = self.split_dec_ic(g0)
        g_all, _ = self.dec(self.in_0, g0)
        fac = self.factors(g_all)
        xrec = self.out_w(fac)
        return xrec, ic_mean, ic_logvar, fac


# correlation coefficient as an alternative (to MSE) loss component
def corr_coef(x_rec, x):
    cc = ((x-x.mean(0))/x.std() * (x_rec-x_rec.mean())/x_rec.std()).sum()
    n = (x.shape[0]-1)*x.shape[1]*x.shape[2]
    return cc/n


def lfads_loss(x_rec, x, ic_mean, ic_logvar, prior_mean, prior_logvar, kl_scale=1, loss_type='CC'):
    # KL - g0
    kl_loss_g0 = kl_gauss_gauss(ic_mean, ic_logvar, prior_mean, prior_logvar)
    kl_loss_g0 /= ic_mean.shape[0]
    kl_loss_g0 *= kl_scale

    if loss_type=='CC':
        reco_loss = -corr_coef(x_rec, x)
    elif loss_type=='MSE':
        mse = nn.MSELoss(reduction='mean')
        reco_loss = mse(x, x_rec)
    elif loss_type=='MSE+CC':
        mse = nn.MSELoss(reduction='mean')
        reco_loss = mse(x, x_rec) - corr_coef(x_rec, x)

    return kl_loss_g0.sum() + reco_loss


def kl_gauss_gauss(z_mean, z_logvar, prior_mean, prior_logvar, varmin=1e-16):
    """Compute the KL divergence."""

    z_logvar_wm = torch.log(z_logvar.exp() + varmin)
    prior_logvar_wm = torch.log(prior_logvar.exp() + varmin)
    return (0.5 * (prior_logvar_wm - z_logvar_wm
             + torch.exp(z_logvar_wm - prior_logvar_wm)
             + torch.pow((z_mean - prior_mean) / torch.exp(0.5 * prior_logvar_wm), 2)
             - 1.0))


def get_kl_warmup_fun(lfads_opt_hps):
    """Ramp up the KL loss during training"""

    kl_warmup_start = lfads_opt_hps['kl_warmup_start']
    kl_warmup_end = lfads_opt_hps['kl_warmup_end']
    kl_min = lfads_opt_hps['kl_min']
    kl_max = lfads_opt_hps['kl_max']
    def kl_warmup(batch_idx):
        progress_frac = ((batch_idx - kl_warmup_start) /
                         (kl_warmup_end - kl_warmup_start))
        kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                             (kl_max - kl_min) * progress_frac + kl_min)
        return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
    return kl_warmup


class SequentialDataset(Dataset):
    def __init__(self, data, transforms=None):
        # time X batch X features
        self.data = data
        # get torchvision transforms ??
        self.transforms = transforms

    def __getitem__(self, idx):
        return self.data[:, idx, :]

    def __len__(self):
        return self.data.shape[1]


def collate_seq(batch):
    return torch.stack(batch, dim=1)

