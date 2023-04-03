import os
import glob
import shutil
import fitsio
from pathlib import Path
import numpy as np
import pandas as pd
import calibrate_jp2
import sunpy.io

# Convert jp2 files at wavelengths relevant to coronal holes: 193 & 211
datadir = os.path.join(os.environ['DATA'], 'SDO/AIA/jp2_data')
globalcsvf = Path(datadir, 'label_jp2_map_global.csv')
labelmap = pd.read_csv(globalcsvf)


# Directory that will contain the curated jp2 files: no mismatch between the year/month folder and the year_month in
# the filename. For example: 2011_01 folder (Januaray) had files from 2011_02_01 (February...)
cleandir = 'jp2_clean'

# jp2fs_193 = sorted(glob.glob(os.path.join(os.environ['DATA'], 'SDO/AIA/jp2_data/2011/*/jp2/*_SDO_AIA_AIA_193.jp2')))
# jp2fs_211 = sorted(glob.glob(os.path.join(os.environ['DATA'], 'SDO/AIA/jp2_data/2011/*/jp2/*_SDO_AIA_AIA_211.jp2')))
# jp2fs = jp2fs_193 + jp2fs_211

year = '2011'
# List all months
months = [f'{i:02d}' for i in range(1, 13)]

for m in months:
    cleanpath = Path(datadir, 'curated', year, m)
    cleanjp2dir = Path(cleanpath, 'jp2')
    cleanlabeldir = Path(cleanpath, 'label')
    cleanfitsdir = Path(cleanpath, 'fits')
    os.makedirs(cleanjp2dir, exist_ok=True)
    os.makedirs(cleanfitsdir, exist_ok=True)
    os.makedirs(cleanlabeldir, exist_ok=True)

    chmap = labelmap[labelmap['npz file'].str.contains('_CH') & labelmap['npz file'].str.contains(f'{year}_{m}')]
    for idx, row in chmap.iterrows():
        # Absolute path to label mask npz file
        npz_abspath = Path(datadir, year, f'{year}_{m}', 'label_masks', row['npz file'])
        # Absolute path to jp2
        jp2_abspath = Path(datadir, year, f'{year}_{m}', 'jp2', row['jp2 AIA 193'])
        print(str(npz_abspath), '----', str(jp2_abspath))
        if not npz_abspath.exists() or not jp2_abspath.exists():
            print('file not found')
            continue
        # New fits path
        fits_newpath = Path(cleanfitsdir, row['jp2 AIA 193']).with_suffix('.fits')
        # New label path
        npz_newpath = Path(cleanlabeldir, row['npz file'])
        if fits_newpath.exists() and npz_newpath.exists():
            print('already calibrated and saved')
            continue
        # Read and calibrate
        cal_data, cal_header = calibrate_jp2.read_sdo_jp2(jp2_abspath, verbose=True)
        # Fix the headers. They contain float "nan" values that make fitsio.write() crash.
        for k, v in cal_header.items():
            if not isinstance(v, str) and np.isnan(v):
                cal_header[k] = str(v)
        # Write calibrated jp2, FITS, and copy npz in same month directory
        sunpy.io.write_file(str(Path(cleanjp2dir, row['jp2 AIA 193'])), cal_data, cal_header)
        fitsio.write(fits_newpath, cal_data, header=cal_header,
                     compress='RICE', clobber=True)
        shutil.copy(npz_abspath, npz_newpath)








        # jp2 = glymur.Jp2k(jp2f)
        # fullres = jp2[:]
        # # Create output FITS directory and file names
        # fits_dir = Path(jp2f.parent.parent, 'fits')
        # fits_f = Path(fits_dir, jp2f.stem+'.fits')
        # os.makedirs(fits_dir, exist_ok=True)
        # # Write to FITS file, restoring original int16 type of level 1 FITS files with RICE compression
        # fitsio.write(fits_f, fullres.astype(np.int16), compress='RICE', clobber=True)
        # print(f'wrote file {i+1}/{nfiles}: {fits_f}')
