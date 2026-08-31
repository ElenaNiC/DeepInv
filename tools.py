
import json
import ast
import numpy as np
from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt
from PIL import Image
import torch


def binArray(data, axis, binstep, binsize, func=np.nanmean):

    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

def load_experiment(exp_dir: str | Path, gr: int, l: int, file_prefix: str = "spectral", ref=True) -> dict:
    exp_dir = Path(exp_dir)
    meta_path = exp_dir / "metadata.json"
    raw_dir = exp_dir / "raw_data"
    overview_dir = exp_dir / "overview"

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    acq = metadata[-1]
    dmd = metadata[0]

    wavelengths = np.array(ast.literal_eval(acq["wavelengths"]), dtype=float)
    Lc = np.array(ast.literal_eval(acq["Lc"]), dtype=float)[0][0]

    patterns = acq["patterns"]
    p_x = acq["pattern_dimension_x"]
    p_y = acq["pattern_dimension_y"]
    M = int(dmd["patterns"])

    # build filenames in numeric order
    files = [
        raw_dir / f"{file_prefix}_NR_0_Gr_{gr}_Lc_{l}nm_NA_0_NS_{k}.npz"
        for k in range(M)
    ]

    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} files, first missing: {missing[0]}"
        )

    first = np.load(files[0], allow_pickle=True)["arr_0"].astype(np.float32)
    N, L = first.shape

    if L != len(wavelengths):
        print(f"Warning: file has L={L} channels, metadata has {len(wavelengths)} wavelengths")

    y = np.empty((M, N, L), dtype=np.float32)
    y[0] = first

    for k, f in enumerate(files[1:], start=1):
        arr = np.load(f, allow_pickle=True)["arr_0"].astype(np.float32)
        if arr.shape != (N, L):
            raise ValueError(f"Shape mismatch in {f.name}: got {arr.shape}, expected {(N, L)}")
        y[k] = arr

    
    raw_data = np.moveaxis(y, 1, 0)
    bin_fact =  N / (M//2)

    if bin_fact == 1:
        spectral_data_all = raw_data.astype(float, copy=True)
    else:
        spectral_data_all = binArray(raw_data, 0, bin_fact, bin_fact)
    

    f = raw_dir / f"spatial_NR_0_Gr_{gr}_Lc_{l}nm_NA_0_NS_1.npz"
    data = np.load(f)

    spatial_data = data[data.files[0]]

    
    #unsplit raw data
    unsplit = raw_data[:,0::2, :] - raw_data[:,1::2, :]
    unsplit_binned = spectral_data_all[:,0::2, :] - spectral_data_all[:,1::2, :]

    if ref:
        bin_img   = iio.imread(overview_dir / "spectral_BIN_IMAGE_had_reco.png")
        gray_img  = iio.imread(overview_dir / "spectral_GRAY_IMAGE_had_reco.png")
        rgb_img   = iio.imread(overview_dir / "spectral_RGB_IMAGE_had_reco.png")
        slice_img = iio.imread(overview_dir / "spectral_SLICE_IMAGE_had_reco.png")
        spectra   = iio.imread(overview_dir / "spectral_SPECTRA_PLOT_had_reco.png")

        return {
            "dir": exp_dir,
            "metadata": metadata,
            "acq": acq,
            "wavelengths": wavelengths,
            "Lc": Lc,
            "patterns": patterns,
            "pattern_dimension_x": p_x,
            "pattern_dimension_y": p_y,
            "M": M,
            "N": N,
            "L": L,
            "y": y,
            "raw_data": raw_data,
            "spectral_data_all": spectral_data_all,
            "unsplit": unsplit,
            "unsplit_binned": unsplit_binned,
            "bin_img": bin_img,
            "gray_img": gray_img,
            "rgb_img": rgb_img,
            "slice_img": slice_img,
            "spectra": spectra,
            "spatial_data": spatial_data,
        }
    else:
             return {
            "dir": exp_dir,
            "metadata": metadata,
            "acq": acq,
            "wavelengths": wavelengths,
            "Lc": Lc,
            "patterns": patterns,
            "pattern_dimension_x": p_x,
            "pattern_dimension_y": p_y,
            "M": M,
            "N": N,
            "L": L,
            "y": y,
            "raw_data": raw_data,
            "spectral_data_all": spectral_data_all,
            "spatial_data": spatial_data,
            "unsplit": unsplit,
            "unsplit_binned": unsplit_binned
             }


def bin_cols(Mat,n):

	(Nr,Nc) = np.shape(Mat)
	M_out = np.zeros((Nr,Nc//n))

	for i in range(0,Nc,n):
		for j in range(n):
			M_out[:,i//n] += Mat[:,i+j]
	return(M_out)



def hadamard(M, w):

    
    pattern_dir = Path(f"C:/Users/ceidigh/Documents/Obair/data/Patterns/Walsh_{M}x{M}")
    fname = pattern_dir / f"Walsh_{M}x{M}_0.png"
    png = np.array(Image.open(fname).convert("L"))
    px,py = png.shape

    A_png = np.zeros((2*M, px, px)) # want square - patterns have border in y

    for i in range(2*M):
            fname = pattern_dir / f"Walsh_{M}x{M}_{i}.png"
            png = np.array(Image.open(fname).convert("L"))
            png = (png > 0).astype(float)
            
            A_png[i] = png[:, ((py-px)//2):(px + ((py-px)//2))]

    px,py = A_png[0].shape
    pat_pos = np.zeros((M,px)) # half measures are positive (1s)
    pat_neg = np.zeros((M,px)) # other half need to be minused (represent the -1s that the DMD can't do)

    for i in range(0,M*2,2):
        pat_pos[i//2,:] =  A_png[i, 400, :] # just select a row - not measured data so all are identical and already in binary
        pat_neg[i//2,:] =  A_png[i+1, 400, :]# just select a row - not measured data so all are identical and already in binary 

        if((A_png[i, 400, :] - pat_pos[i//2,:]).sum() != 0 ):
            print("Check failed at i = ",i, ":", A_png[i, 400, :] - pat_pos[i//2,:])

        if((A_png[i+1, 400, :] - pat_neg[i//2,:]).sum() != 0 ):
            print("Check failed at i = ",i, ":", A_png[i, 400, :] - pat_pos[i//2,:])


    # make square
    pat_pos = bin_cols(pat_pos,  (px // (w))) 
    pat_neg = bin_cols(pat_neg,  (px // (w)))
    H = pat_pos-pat_neg

    #normalise
    H = H//(px // (w))

    return pat_pos, pat_neg, H
