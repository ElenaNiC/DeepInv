# %% imports

import os
import torchvision
import torch.nn
import numpy as np

import matplotlib.pyplot as plt
from spyrit.misc.disp import imagesc
from spyrit.misc.statistics import transform_gray_norm, data_loaders_stl10

from metadata_SPC2D import read_metadata
from spyrit.misc.walsh_hadamard import walsh_matrix_2d

# %%  Download Data 
from pathlib import Path
from spyrit.misc.load_data import download_girder

destination = Path("C:/Users/ceidigh/Documents/DeepInv/") 
print("Copying folder in:", destination)

url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
data_subfolder = Path("data")
data_files = [

    "61e19b42cdb6910b899d0150",  # SeimensStar spectraldata.npz
    "61e19b40cdb6910b899d014d",  # SeimensStar metadata.json
]
try:
    download_girder(url_pilot, data_files, data_subfolder)
except Exception as e:
    print("Unable to download data from the Pilot warehouse")
    print(e)


# %% Cube Reconstruction  - all channels

#  Get Cubes
raw_data = np.load("./data/SeimensStar_whiteLamp_linear_color_filter_spectraldata.npz", allow_pickle=True)["spectral_data"]
P, L = raw_data.shape

#Get pattern order
meta_path = "./data/SeimensStar_whiteLamp_linear_color_filter_metadata.json"
_, acquisition_parameters, _, _ = read_metadata(meta_path)
wavelengths = acquisition_parameters.wavelengths 
N = acquisition_parameters.pattern_dimension_x  # probably np.sqrt(P//2) for uncompressed sensing (? right)

patterns = acquisition_parameters.patterns # order of acquisition patterns
ind = np.array(patterns[0::2])//2 # corresponding hadamard rows


#%%  prep raw data - unsplit and reorder
m_unsplit = raw_data[0::2, :] - raw_data[1::2, :]
m_reordered = np.zeros((N*N, L)) # create space first
m_reordered[ind, :]  = m_unsplit # place rows of m_unsplit following order of ind :)

# bin y along spectral dimension frist -> 2048 is too slow
# takes a while so maybe just save and load
from tools import binArray
m_reordered.shape
m_reordered = binArray(m_reordered, -1, 4, 4, np.sum)
m_reordered.shape

np.save("m.npy", m_reordered,)
#%% LOAD PRE FORMATTED DATA

m = np.load("m.npy")
Nx, L = m.shape
m.shape
#%%
# direct reconstruction 
H = walsh_matrix_2d(N)   #fix 
f = np.matmul(H,m)
f = np.reshape(f, (N,N, L))// (N*N)

plt.imshow(f.sum(-1)) # plot wavelength sum

# %% Pre-summed (single channel) reconstruction 

#  Get Cubes
raw_data = np.load("./data/SeimensStar_whiteLamp_linear_color_filter_spectraldata.npz", allow_pickle=True)["spectral_data"]
raw_data = raw_data.sum(-1)
P = raw_data.shape

#Get pattern order
meta_path = "./data/SeimensStar_whiteLamp_linear_color_filter_metadata.json"
_, acquisition_parameters, _, _ = read_metadata(meta_path)
wavelengths = acquisition_parameters.wavelengths 
N = acquisition_parameters.pattern_dimension_x  # probably np.sqrt(P//2) for uncompressed sensing (? right)

patterns = acquisition_parameters.patterns # order of acquisition patterns
ind = np.array(patterns[0::2])//2 # corresponding hadamard rows


# prep raw data - unsplit and reorder
m_unsplit = raw_data[0::2] - raw_data[1::2]
m_reordered = np.zeros((N*N)) # create space first
m_reordered[ind]  = m_unsplit # place rows of m_unsplit following order of ind :)

# direct reconstruction
H = walsh_matrix_2d(N)
f = np.matmul(H,m_reordered)
f_presum= np.reshape(f, (N,N))// (N*N)

plt.imshow(f_presum) # plot wavelength sum
plt.title("presummed Direct Reconstruction")

# %% now try with deepinv
import deepinv as dinv
from spyrit.core.meas import HadamSplit2d
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# define a spyrit meas_op
meas_spyrit = HadamSplit2d(N, device=device, reshape_output=True)
# calculer parametre de normalisation
norm = torch.linalg.norm(meas_spyrit.H, ord=2)
print(norm)

m_reordered = m 
m_reordered = torch.from_numpy(m_reordered).to(device)
# %%  create corresponding deepinv meas_op

meas_deepinv = dinv.physics.LinearPhysics(
    lambda y: meas_spyrit.measure_H(y) / norm,
    A_adjoint=lambda y: meas_spyrit.unvectorize(meas_spyrit.adjoint_H(y) / norm),
)

m_reordered = m_reordered.reshape(m_reordered.shape[-1], 1, m_reordered.shape[0] ) #put spectral channel in batch 



x_pinv = meas_deepinv.A_dagger(m_reordered.to(torch.float32) / norm)
imagesc(x_pinv[100, 0,:,:].detach().cpu(), "presummed direct reconstruction using deepinv ")

# %% AND THEN TRY WITH RAM 


model_ram = dinv.models.RAM(pretrained=True, device=device)
model_ram.sigma_threshold = 1e-1

y = m_reordered.to(torch.float32) 
y = y.view(1, 1, m_reordered.shape[0] )

with torch.no_grad():
    x_ram = model_ram(y / norm, meas_deepinv)
imagesc(x_ram[0, 0, :, :].cpu(), "RAM recon")



# %%


physics = meas_deepinv
