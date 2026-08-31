

# for each measurement file: 
# read correspondning meta_Data pattern list
# unsplit and reorder measurements according to pattern list
# bin spectrally and subsample measurement wise


# example for one file:
#  Get Cubes
raw_data = np.load("./ground_truth/SeimensStar_whiteLamp_linear_color_filter_spectraldata.npz", allow_pickle=True)["spectral_data"]
P, L = raw_data.shape

#Get pattern order
meta_path = "./metadata/SeimensStar_whiteLamp_linear_color_filter_metadata.json"
_, acquisition_parameters, _, _ = read_metadata(meta_path)
wavelengths = acquisition_parameters.wavelengths 
N = acquisition_parameters.pattern_dimension_x  # probably np.sqrt(P//2) for uncompressed sensing (? right)

patterns = acquisition_parameters.patterns # order of acquisition patterns
ind = np.array(patterns[0::2])//2 # corresponding hadamard rows

m_unsplit = raw_data[0::2, :] - raw_data[1::2, :]
m_reordered = np.zeros((N*N, L)) # create space first
m_reordered[ind, :]  = m_unsplit # place rows of m_unsplit following order of ind :)

# bin y along spectral dimension frist -> 2048 is too slow
# takes a while so maybe just save and load
from tools import binArray
m_reordered.shape
m_reordered = binArray(m_reordered, -1, 4, 4, np.sum)
m_reordered.shape

K = 128 # or whatever number of measures you want to use
m = m_reordered[0:K, :]

# for each GT file
# bin spectrally