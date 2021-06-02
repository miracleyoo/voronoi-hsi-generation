import pandas as pd
import numpy as np
import random
import math

# This will make sure the dataframe returned is not erronous


def read_df_from_txt(file_path):
    df = pd.read_table(file_path, sep='\t', skiprows=2, names=[
                       'wavelength', 'reflectance', 'std'])

    #     pd.to_numeric(df['wavelength'], errors='coerce', downcast='float')
    #     pd.to_numeric(df['reflectance'], errors='coerce', downcast='float')
    #     pd.to_numeric(df['std'], errors='coerce', downcast='float')

    for index, row in df.iterrows():
        try:
            row['wavelength'] = float(row['wavelength'])
            row['reflectance'] = float(row['reflectance'])
            row['std'] = float(row['std'])
        except Exception:
            #             print(f'Dropped row: \n{row} \n')
            df.drop(index, inplace=True)

    # df.astype('int64').dtypes

    return df


def resample(df: pd.DataFrame, n=5):
    """
    Resample data to be n steps
    """
    # df2 = df.groupby(df.index // n).mean()
    # df2.index *= n

    Xresampled = np.arange(df.first_valid_index(), df.last_valid_index()+n, n)
    df_resampled = df.reindex(df.index.union(
        Xresampled)).interpolate('values').loc[Xresampled]
    return df_resampled


# Preprocess wavelength data
# Resampling the wavelength into steps of n
# Investigate weird errors
def preprocess_material_response(df, n=5, start=300, limit=3000, start_threshold=50, end_threshold=500,
                                 ignore_limits=False):
    # Assuming the format is df" [wavelength, reflectance]

    # We first convert the wavelength from micron to nanometer
    df['wavelength'] = df['wavelength'].apply(lambda x: int(x * 1000))

    # Group by wavelength to use wavelength as index
    df = df.groupby('wavelength').mean()

    # Here we make sure that the steps of the wevelength are of step n
    df2 = resample(df, n)

    # To reduce runtime, we can limit the range we are evaluating to
    # This limit should be set so it is the same as the limit of the camera response function
    # We want to make sure that the start and end is within the boundaries that we set

    ori_fvi = df2.first_valid_index()
    ori_lvi = df2.last_valid_index()

    df2 = df2.loc[df2.index >= start]
    df2 = df2.loc[df2.index <= limit]

    if ignore_limits == False:
        fvi = df2.first_valid_index()
        lvi = df2.last_valid_index()

        if fvi is None or lvi is None:
            raise ValueError(
                f'The start ({start}) and limit ({limit}) values yielded an empty dataframe. The material wavelength started at {ori_fvi} and ends at {ori_lvi}')

        if abs(fvi - start) > start_threshold:
            raise ValueError(
                f'The material wavelength starts at {fvi} while our starting value is {start} with a threshold of {start_threshold}')

        if abs(lvi - limit) > end_threshold:
            raise ValueError(
                f'The material wavelength ends at {lvi} while our ending value is {limit} with a threshold of {end_threshold}')

    return df2


def get_weighted_sum(material_df, camera_df, band_name):
    weight_sum = 0
    missing_idx = set()

    for wv, row in material_df.iterrows():
        try:
            if camera_df.at[wv, band_name] > 0:
                weight_sum += (row['reflectance'] *
                               camera_df.at[wv, band_name])
        #                 print(f"Wavelength {wv} : {row['reflectance']} * {df1.at[wv, band_name]}")
        except KeyError:
            #            print(f'Key {wv} not found')
            missing_idx.add(wv)
            continue

    #     print(f'Missing wavelength: {missing_idx} with length {len(missing_idx)}')
    return weight_sum, missing_idx


# Gets weighted sum from processed material and camera dataframe
def get_weighted_sums_from_df(material_df, camera_df, highlight_missing=False):
    weighted_sums = []
    for band_name in camera_df.columns:
        wsum, missing = get_weighted_sum(material_df, camera_df, band_name)
        weighted_sums.append(wsum)

    #     print(f'There are {len(missing)} missing wavelengths in the material function')

    # # Normalize the array
    # norm = np.linalg.norm(weighted_sums)

    # weighted_sums = weighted_sums / norm

    if highlight_missing:
        print(f'Missing wavelength: {missing} with length {len(missing)}')

    return missing, weighted_sums


def get_weighted_sums_from_txt_file(txt_file, camera_df, config={}):
    df = read_df_from_txt(txt_file)
    df = preprocess_material_response(df, n=config['step'], start=config['start_wavelength'], limit=config['end_wavelength'], start_threshold=config['start_threshold'],
                                      end_threshold=config['end_threshold'], ignore_limits=config['ignore_limits'])

    return get_weighted_sums_from_df(df, camera_df)


def get_random_success_files(n=25, pkl='success.pkl'):
    success_df = pd.read_pickle(pkl)
    sampled_df = success_df.sample(n)

    return np.array(sampled_df['file_path'])


def get_random_weighted_sums(n=25, pkl='success.pkl', dim=None):
    """
    Get the weighted sums of n materials from pkl satelite response functions
    @params:
        n   (int)              :  number of random weighted sums to be generated
        pkl (pkl: dataframe)   :  pickle file of camera function in dataframe
        dim ((int,int))        :  dimension of returned array in (height,width)
    """
    camera_df = pd.read_pickle(pkl)
    random_files = get_random_success_files(n, pkl)
    for i, row in enumerate(random_files):
        for j, el in enumerate(row):
            random_files[i][j] = get_weighted_sums_from_txt_file(el, camera_df)

    if dim is not None:
        random_files = np.array(random_files).reshape(dim)

    return random_files


def reshape_array(arr, h, w):
    return np.array(arr).reshape((h, w))


# Growing Algorithm Helper Functions
def voronoi(width, height, num_cells, materials_array, band_num=13):
    """
    Generates a voronoi diagaram with num_cells patches using materials from materials array
    @params:
        width   (int)                   :  width of the image
        height   (int)                  :  height of the image
        num_cells   (int)               :  number of points/patches generated
        materials_array(str[])          :  array containing the txt files of the materials
    """
    img = np.zeros((width, height, band_num))
    imgx, imgy = width, height

    nx = []
    ny = []
    mat_choice = []

    for i in range(num_cells):
        # Append random points
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))

        # Append their corresponding rgb values
        mat_choice.append(
            materials_array[random.randrange(len(materials_array))])

    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i
            img[x][y] = mat_choice[j]

    return img
