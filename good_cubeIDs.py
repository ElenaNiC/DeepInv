
from pathlib import Path
from spyrit.misc.load_data import download_girder

# %%
# %% SETTINGS
# change this if you want to download the files in a different folder
destination = Path("C:/Users/ceidigh/Documents/DeepInv/") 
print("Copying folder in:", destination)


# %%
# download data from the Pilot warehouse
url_pilot = "https://pilot-warehouse.creatis.insa-lyon.fr/api/v1"
data_subfolder = Path("data")
data_files = [

    "61e19b42cdb6910b899d0150",  # SeimensStar spectraldata.npz
    "61e19b40cdb6910b899d014d",  # SeimensStar metadata.json

# setup_v1.3.1 test_HCERES
    "69120e50c68404167c562b34", # obj_Cat_bicolor_thin_overlap_source_white_LED_Walsh_im_64x64_ti_9ms_zoom_x1 spectraldata.npz
    "69120e4dc68404167c562b2e", # metadata
    "69120e4fc68404167c562b31", # nn recon
    "69120e4cc68404167c562b2b", # had recon

# SPIHIMdata setup_v1.3.1 2024-07-19_spectral_unmixing

    "669a62197c2d35c7ba3b445d", # obj_RGB_geometrical_form_trans_source_white_LED_Walsh_im_64x64_ti_2ms_zoom_x1_spectraldata.npz
    "669a62187c2d35c7ba3b445a", # metadata
    "669a62187c2d35c7ba3b4457", # had recon

    "669a2b4a7c2d35c7ba3b437d", # obj_RGB_geometrical_form_source_Thorlabs_White_halogen_lamp_Walsh_im_64x64_ti_40ms_zoom_x1_spectraldata.npz
    "669a2b497c2d35c7ba3b437a", # metadata
    "669a2b497c2d35c7ba3b4377", # had recon

    "669a765f7c2d35c7ba3b4625", # obj_RGB_geometrical_form_reflexion_source_white_LED_Walsh_im_64x64_ti_50ms_zoom_x1_spectraldata.npz
    "669a765f7c2d35c7ba3b4622", # metadata
    "669a765f7c2d35c7ba3b461f", # had recon
    
    "669a87847c2d35c7ba3b4715", # obj_CMY_geometrical_form_reflexion_source_white_LED_Walsh_im_64x64_ti_50ms_zoom_x1_spectraldata.npz
    "669a87847c2d35c7ba3b4712", # metadata
    "669a87837c2d35c7ba3b470f", # had recon

# setup_v1.3.1 2022-11-16_Apple
    "6375166e4d15dd536f0482b3", # Bitten_Apple_t_20min-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms_spectraldata.npz
    "6375166d4d15dd536f0482b0", # metadata
    "6375166d4d15dd536f0482ad", # had recon
    "645403f185f48d3da071785c", # nn recon

    "63750b6b4d15dd536f0481a9", # Apple_pos_1-im_64x64_Zoom_x1_ti_10ms_tc_0.5ms_spectraldata.npz
    "63750b6a4d15dd536f0481a6", # metadata
    "63750b6a4d15dd536f0481a3", # had recon
    "6453ee6e85f48d3da0717736", # nn recon

# setup_v1.3.1 2022-12-09_color_checker
    "6393327a4d15dd536f04839a", # color_checker_full_FOV_64x64_Zoom_x1_ti_15ms_tc_0.2ms_spectraldata.npz
    "639332794d15dd536f048397", # metadata
    "639332794d15dd536f048394", # had recon
    "64540ea685f48d3da07178ef", # nn recon

# setup_v1.3.12023-03-07_Arduino_hologram
    "640741290386da2747650c33", # full_hologram_2d_angle_ill128x128_ti_10ms_zoom_x1_spectraldata.npz
    "640741280386da2747650c30", # metadata
    "640741270386da2747650c2d", # had recon

    "640722cc0386da2747650c13", # full_hologram64x64_ti_10ms_zoom_x1_spectraldata.npz
    "640722cc0386da2747650c10", # metadata
    "640722cb0386da2747650c0d", # had recon
    "64544a6385f48d3da0717b3b", # nn recon

# setup_v1.3.1 2023-03-08_Arduino_hologram
    "6408a05e0386da2747650c74", # full_hologram_90Deg64x64_ti_5ms_zoom_x1_spectraldata.npz
    "6408a05e0386da2747650c71", # metadata
    "6408a05a0386da2747650c6e", # had recon
    "6454c1c385f48d3da0717d10", # nn recon

    "64089c9b0386da2747650c54", # full_hologram64x64_ti_5ms_zoom_x1_spectraldata.npz
    "64089c860386da2747650c51", # metadata
    "64089c860386da2747650c4e", # had recon
    "6454bdcf85f48d3da0717b67", # nn recon

# setup_v1.3.1 2023-03-08_banknote
    "6408bdc80386da2747650d95", # 20€_banknote_Back_of_PrincessEurope_ill_angle_3_64x64_ti_10ms_zoom_x1_spectraldata.npz
    "6408bdb40386da2747650d92", # metadaata
    "6408bdb10386da2747650d8f", # had recon
    "6454d88185f48d3da0717fa5", # nn recon

# setup_v1.3.1 2024-06-06_bacteriocyte
    "6662e19f7c2d35c7ba3b2f4a", # obj_bacteriocyte_source_Thorlabs_White_halogen_lamp_f80mm-P2_Walsh_im_64x64_ti_400ms_zoom_x4_spectraldata.npz
    "6662e19a7c2d35c7ba3b2f44", # metadata
    "6662e19a7c2d35c7ba3b2f41", # had recon
    "6662e19d7c2d35c7ba3b2f47", # nn recon

    "6662e1a67c2d35c7ba3b2f7c", # obj_bacteriocyte_trans_cluster_centered_source_Thorlabs_White_halogen_lamp_f80mm-P2_Walsh_im_64x64_ti_50ms_zoom_x4_spectraldata.npz
    "6662e1a27c2d35c7ba3b2f76", # metadata
    "6662e1a27c2d35c7ba3b2f73", # had recon
    "6662e1a57c2d35c7ba3b2f79", # nn recon

    "6662e1ae7c2d35c7ba3b2fae", # obj_bacteriocyte_trans_source_Thorlabs_White_halogen_lamp_f80mm-P2_Walsh_im_64x64_ti_60ms_zoom_x4_spectraldata.npz
    "6662e1aa7c2d35c7ba3b2fa8", # metadata
    "6662e1a97c2d35c7ba3b2fa5", # had recon
    "6662e1ad7c2d35c7ba3b2fab", # nn recon

# setup_v1.3.1 2026-08-27_freeform_publication
    "6a900548f5d51d66558ecb91", # obj_cat_source_white_LED_hadam2d_32768_im_128x128_ti_2.5ms_zoom_x1_spectraldata.npz
    "6a900542f5d51d66558ecb8d", # metadata
    "6a900542f5d51d66558ecb89", # had recon

# setup_v1.3 2022-03-11_Cat Cat_LinearColoredFilter
    "622b5eaa43258e76eab2174f", # Cat_LinearColoredFilter_spectraldata.npz
    "622b5ea943258e76eab2174c", # metadata
    "622b5ea943258e76eab21749", # had recon
    "644ea26585f48d3da0716958", # nn recon

# setup_v1.3 2022-03-11_Horse Horse_LinearColoredFilter
    "622b638b43258e76eab2179c", # Horse_LinearColoredFilter_spectraldata.npz
    "622b638943258e76eab21799", # metadata
    "622b638943258e76eab21796", # had recon
    "644eab3785f48d3da0716a1a", # nn recon

# setup_v1.32022-01-14_SeimensStar_LinearColorFilterSeimensStar_whiteLamp_linear_color_filter
    "61e19b42cdb6910b899d0150", # SeimensStar_whiteLamp_linear_color_filter_spectraldata.npz
    "61e19b40cdb6910b899d014d", # metadata
    "61e19b40cdb6910b899d014a", # had recon
    "644dc81985f48d3da0716805", # nn recon

# setup_v1.3 2022-09-19_color_checker
    "63288a96ebe9129ae9936f37", # color_checker_ti_1ms_without_telecentric_2_spectraldata.npz
    "63288a95ebe9129ae9936f34", # metadata
    "63288a94ebe9129ae9936f31", # had recon
    "6450a62b85f48d3da0716c2d", # nn recon

# setup_v1.3 2022-09-19_Thorlabs_box
    "63289430ebe9129ae9936f99", # Thorlabs_box_ti_10ms_without_telecentric_spectraldata.npz
    "63289430ebe9129ae9936f96", # metadata
    "6328942febe9129ae9936f93", # had recon
    "644a4d9a85f48d3da0714635", # nn recon

# setup_v1.3 2022-09-20_Blob
    "6329d772ebe9129ae99370fa", # Blob_ti_60ms_zoomx6_objx10_spectraldata.npz
    "6329d771ebe9129ae99370f7", # metadata
    "6329d771ebe9129ae99370f4", # had recon
    "6450e6db85f48d3da0716f4a", # nn recon

    "6329dc0febe9129ae993711a", # Blob_ti_120ms_zoomx12_objx10_spectraldata.npz
    "6329dc0eebe9129ae9937117", # metadata
    "6329dc0eebe9129ae9937114", # had recon
    "6450c7c285f48d3da0716db1", # nn recon

# setup_v1.3 2022-09-21_vegetals
    "632b2b6eebe9129ae993727c", # green_Tree_leaf_ti_10ms_zoomx4_objx40_spectraldata.npz
    "632b2b6debe9129ae9937279", # metadata
    "632b2b6debe9129ae9937276", # had recon
    "6451692d85f48d3da07170b1", # nn recon

    "632b2adcebe9129ae993725c", # green_Tree_leaf_ti_20ms_zoomx4_objx40_spectraldata.npz
    "632b2adbebe9129ae9937259", # metadata
    "632b2adbebe9129ae9937256", # had recon
    "6451752f85f48d3da0717113", # nn recon

# setup_v1.3 2022-09-29_test_different_image_size NO SPECTRA PLOT :(
    "6335ba75ebe9129ae993e51b", # Star_Sector_test_+4_rounds_image_size_128x128_zoom_x6_spectraldata.npz
    "6335ba72ebe9129ae993e518", # metadata
    "6335ba72ebe9129ae993e515", # had recon

]
try:
    download_girder(url_pilot, data_files, data_subfolder)
except Exception as e:
    print("Unable to download data from the Pilot warehouse")
    print(e)

# %%
