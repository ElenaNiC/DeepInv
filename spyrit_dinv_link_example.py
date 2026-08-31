
#%% imports
import os
import torchvision
import torch.nn

import matplotlib.pyplot as plt
from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm, data_loaders_stl10
import deepinv as dinv

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %% Download / access images
# Grayscale images of size (32, 32), no normalization to keep values in (0,1)
transform = transform_gray_norm(img_size=32, normalize=False)

# Create dataset and loader (expects class folder 'images/test/')
img_size = 32
batch_size = 7
data_root = 'C:/Users/ceidigh/Documents/PFE/Code/data'
dataloader = data_loaders_stl10(data_root, img_size=img_size, batch_size=batch_size, seed=7, shuffle=False, download=True, normalize=False)


x, _ = next(iter(dataloader['train']))
print(f"Ground-truth images: {x.shape}")
# %% Plot second image
i_plot = 1
imagesc(x[i_plot, 0, :, :], r"$32\times 32$ image $X$")

# %% Simulate 2d hadamard acquisition -> K = 512 (< N = 32^2 = 1024) 

from spyrit.core.meas import HadamSplit2d
import spyrit.core.noise as noise
from spyrit.core.prep import UnsplitRescale

x = x.to(device)
# generate a 32x32 x 32x32 Hadamard, take 512 rows
meas_spyrit = HadamSplit2d(32, 512, device=device, reshape_output=True)
alpha = 50  # image intensity
meas_spyrit.noise_model = noise.Poisson(alpha)
y = meas_spyrit(x)

# preprocess
prep = UnsplitRescale(alpha)
m_spyrit = prep(y)

print(y.shape)
# %% calculer parametre de normalisation
norm = torch.linalg.norm(meas_spyrit.H, ord=2)
print(norm)
# %% Convertir le meas_op spyrit en meas_op deep_inv


# donne lui l'operator 'A' -> qui est 
meas_deepinv = dinv.physics.LinearPhysics(
    lambda y: meas_spyrit.measure_H(y) / norm,
    A_adjoint=lambda y: meas_spyrit.unvectorize(meas_spyrit.adjoint_H(y) / norm),
)
# meas_deepinv.noise_model = dinv.physics.GaussianNoise(sigma=0.01)
m_deepinv = meas_deepinv(x)
print("diff:", torch.linalg.norm(m_spyrit / norm - m_deepinv))
# %% Adjoint and Pseudoinverse reconstructions
x_adj = meas_deepinv.A_adjoint(m_spyrit / norm)
imagesc(x_adj[1, 0, :, :].cpu(), "Adjoint")

x_pinv = meas_deepinv.A_dagger(m_spyrit / norm)
imagesc(x_pinv[1, 0, :, :].cpu(), "Pinv")
# %% optimation based: TV initialisé avec pseudoinv

model_tv = dinv.optim.optim_builder(
    iteration="PGD",
    prior=dinv.optim.TVPrior(),
    data_fidelity=dinv.optim.L2(),
    params_algo={"stepsize": 1, "lambda": 5e-2},
    max_iter=10,
    custom_init=lambda y, Physics: {"est": (Physics.A_dagger(y),)},
)

x_tv, metrics_TV = model_tv(m_spyrit / norm, meas_deepinv, compute_metrics=True, x_gt=x)
dinv.utils.plot_curves(metrics_TV)
imagesc(x_tv[1, 0, :, :].cpu(), "TV recon")
# %% Deep plug and play avec NN de débruitage prétrainer - DRUNet

denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, device=device)
model_dpir = dinv.optim.DPIR(sigma=1e-1, device=device, denoiser=denoiser)
model_dpir.custom_init = lambda y, Physics: {"est": (Physics.A_dagger(y),)}
with torch.no_grad():
    x_dpir = model_dpir(m_spyrit / norm, meas_deepinv)
imagesc(x_dpir[1, 0, :, :].cpu(), "DIPR recon")

# %% RAM
model_ram = dinv.models.RAM(pretrained=True, device=device)
model_ram.sigma_threshold = 1e-1
with torch.no_grad():
    x_ram = model_ram(m_spyrit / norm, meas_deepinv)
imagesc(x_ram[1, 0, :, :].cpu(), "RAM recon")

# %%
