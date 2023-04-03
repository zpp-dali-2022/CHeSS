import numpy as np
from astropy import time
from astropy import units as u
import sunpy.io
import warnings
import cv2
import io
import requests
from bs4 import BeautifulSoup
import pandas as pd


class AIAEffectiveArea:

    def __init__(self, url='https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response/', filename=None):
        """
        :param url: the online location of the response table
        :param filename: string optional location of a local response table to read, overrides url
        Usage
        aia_effective_area = AIAEffectiveArea()
        effective_area_ratio = aia_effective_area.effective_area_ratio(171, '2010-10-24 15:00:00')
        """

        # Local input possible else fetch response table file from GSFC mirror of SSW
        if filename is not None:
            response_table = filename
        else:
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            all_versions = [node.get('href') for node in soup.find_all('a') if
                            node.get('href').endswith('_response_table.txt')]
            latest_table_url = url + sorted([table_files for table_files in all_versions if
                                             table_files.startswith('aia_V')])[-1]
            tbl = requests.get(latest_table_url).content
            response_table = io.StringIO(tbl.decode('utf-8'))

        # Read in response table
        self.response_table = pd.read_csv(response_table, sep='\s+', parse_dates=[1], infer_datetime_format=True, index_col=1)

    def effective_area(self, wavelength, time):
        '''
        :param wavelength: float wavelength of the aia target image
        :param time: string in a format to be read by pandas.to_datetime; the time of the aia target image
        :return: the effective area of the AIA detector interpolated to the target_time
        '''

        eff_area_series = self._parse_series(wavelength.value)

        if (pd.to_datetime(time) - eff_area_series.index[0]) < pd.Timedelta(0):
            warnings.warn('The target time requested is before the beginning of AIA', UserWarning)

        return time_interpolate(eff_area_series, time)

    def effective_area_ratio(self, wavelength, time):
        """
        :param wavelength: float wavelength of the aia target image
        :param time: string in a format to be read by pandas.to_datetime; the time of the aia target image
        :return: the ratio of the current effective area to the  pre-launch effective area
        """

        eff_area_series = self._parse_series(wavelength.value)

        launch_value = eff_area_series[eff_area_series.index.min()]

        if (pd.to_datetime(time) - eff_area_series.index[0]) < pd.Timedelta(0):
            warnings.warn('The target time requested is before the beginning of AIA', UserWarning)

        return time_interpolate(eff_area_series, time) / launch_value

    def _parse_series(self, wavelength):

        # Parse the input response table and return a pd.Series

        eff_area_series = self.response_table[self.response_table.WAVELNTH == wavelength].EFF_AREA

        current_estimate = eff_area_series[eff_area_series.index.max()]

        # Add in a distant future date to keep the interpolation flat
        eff_area_series = eff_area_series.reindex(pd.to_datetime(list(eff_area_series.index.values) +
                                                                 [pd.to_datetime('2040-05-01 00:00:00.000')]))
        eff_area_series[-1] = current_estimate

        return eff_area_series


def time_interpolate(ts, target_time):
    """
    :param ts: pandas.Series object with a time index
    :param target_time: string in a format to be read by pandas.to_datetime; the time to be interpolated to
    :return: value of the same type as in ts interpolated at the target_time
    Usage
    new_value = time_interpolate(time_series, '2010-10-24 15:00:00')
    """
    ts1 = ts.sort_index()
    b = (ts1.index > target_time).argmax()  # index of first entry after target
    s = ts1.iloc[b-1:b+1]

    # Insert empty value at target time.
    s = s.reindex(pd.to_datetime(list(s.index.values) + [pd.to_datetime(target_time)]))

    # Linear interpolation is the most logical
    return s.interpolate('time').loc[target_time]


def scale_rotate(image, angle=0, scale_factor=1, reference_pixel=None):
    """
    Perform scaled rotation.
    The output is a padded image that holds the entire rotated image, recentered around the reference pixel.
    Positive-angle rotation rotates image clockwise if the array origin (0,0) map to the bottom left of the image,
    and counterclockwise if the array origin map to the top left of the image.
    :param image: Numpy 2D array
    :param angle: rotation angle in degrees. Positive-angle rotation rotates image clockwise if the array origin (0,0)
    map to the bottom left of the image, and counterclockwise if the array origin map to the top left of the image.
    :param scale_factor: ratio of the wavelength-dependent pixel scale over the target scale of 0.6 arcsec
    :param reference_pixel: tuple of (x, y) coordinate. Given as (x, y) = (col, row) and not (row, col).
    :return: padded scaled and rotated image
    """
    array_center = (np.array(image.shape)[::-1] - 1) / 2.0

    if reference_pixel is None:
        reference_pixel = array_center

    # convert angle to radian
    angler = angle * np.pi / 180
    # Get basic rotation matrix to calculate initial padding extent
    rmatrix = np.matrix([[np.cos(angler), -np.sin(angler)],
                         [np.sin(angler), np.cos(angler)]])

    extent = np.max(np.abs(np.vstack((image.shape * rmatrix,
                                      image.shape * rmatrix.T))), axis=0)

    # Calculate the needed padding or unpadding
    diff = np.asarray(np.ceil((extent - image.shape) / 2), dtype=int).ravel()
    diff2 = np.max(np.abs(reference_pixel - array_center)) + 1
    # Pad the image array
    pad_x = int(np.ceil(np.max((diff[1], 0)) + diff2))
    pad_y = int(np.ceil(np.max((diff[0], 0)) + diff2))

    padded_reference_pixel = reference_pixel + np.array([pad_x, pad_y])
    # padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=(0, 0))
    padded_image = aia_pad(image, pad_x, pad_y)
    padded_array_center = (np.array(padded_image.shape)[::-1] - 1) / 2.0

    # Get scaled rotation matrix accounting for padding
    rmatrix_cv = cv2.getRotationMatrix2D((padded_reference_pixel[0], padded_reference_pixel[1]), angle, scale_factor)
    # Adding extra shift to recenter:
    # move image so the reference pixel aligns with the center of the padded array
    shift = padded_array_center - padded_reference_pixel
    rmatrix_cv[0, 2] += shift[0]
    rmatrix_cv[1, 2] += shift[1]
    # Do the scaled rotation with opencv. ~20x faster than Sunpy's map.rotate()
    rotated_image = cv2.warpAffine(padded_image, rmatrix_cv, padded_image.shape, cv2.INTER_CUBIC)

    return rotated_image


def aia_pad(image, pad_x, pad_y):
    newsize = [image.shape[0]+2*pad_y, image.shape[1]+2*pad_x]
    pimage = np.empty(newsize)
    pimage[0:pad_y,:] = 0
    pimage[:,0:pad_x]=0
    pimage[pad_y+image.shape[0]:, :] = 0
    pimage[:, pad_x+image.shape[1]:] = 0
    pimage[pad_y:image.shape[0]+pad_y, pad_x:image.shape[1]+pad_x] = image
    return pimage


def read_sdo_jp2(filepath, verbose=False):
    """
    :param filepath: The full file path of the SDO .jp2 image to be read in
    :param verbose: Boolean, if True will print status statements
    :return: numpy array of prepped image
    """
    # Read the image and header
    img = sunpy.io.read_file(filepath, filetype='jp2')[0]
    prepped_header = img.header
    # The aia image size is fixed by the size of the detector. For AIA raw data, this has no reason to change.
    aia_image_size = 4096
    # Rotation of image to get vertical y-axis Top-to-Bottom parallel to Solar North-to-South axis.
    if img.header['CROTA2'] != 0:
        if verbose:
            print('Rotating image to solar north')
        prepped_data = scale_rotate(img.data, img.header['CROTA2'])
        prepped_header['CROTA2'] = 0

        center = ((np.array(prepped_data.shape) - 1) / 2.0).astype(int)
        half_size = int(aia_image_size / 2)
        prepped_data = prepped_data[center[1] - half_size:center[1] + half_size, center[0] - half_size:center[0] + half_size].astype(np.float64)

    else:
        prepped_data = img.data.astype(np.float64)

    # Normalizing the image intensity to levels at the start of the mission for AIA
    if 'AIA' in img.header['INSTRUME']:
        if verbose:
            print('Correcting for CCD degradation')
        # initialize effective area class to avoid rereading the calibration table
        aia_effective_area = AIAEffectiveArea()

        prepped_data *= (1./aia_effective_area.effective_area_ratio(img.header['WAVELNTH']*u.AA, time.Time(img.header['DATE-OBS']).to_datetime()))
        prepped_data[prepped_data < 0] = 0
        prepped_header['DATAMIN'] = 0

    # User Warning if there is significant amount of missing data
    non_zero = np.count_nonzero(prepped_data != 0)

    if img.header['WAVELNTH'] in [94, 131, 171, 193, 211, 304, 335]:
        if (prepped_data.size - non_zero) / non_zero > 0.6:
            warnings.warn('Significant amount of data missing in the image', UserWarning)
    else:
        if (prepped_data.size - non_zero) / non_zero > 1.2:
            warnings.warn('Significant amount of data missing in the image', UserWarning)

    return prepped_data, prepped_header


def write_solar_jp2(fname, image_array, image_header):

    sunpy.io.write_file(fname, image_array, image_header, filetype='auto')