import pandas as pd
import numpy as np
import random
import math
import cv2
import os
from skimage import io
from sklearn.cluster import KMeans

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
    # df_resampled = df.groupby(df.index // n).mean()
    # df_resampled.index *= n

    Xresampled = np.arange(df.first_valid_index(), df.last_valid_index()+n, n)
    df_resampled = df.reindex(df.index.union(
        Xresampled)).interpolate('values').loc[Xresampled]
    # print(df_resampled.head())
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


def get_new_dim(img, long_edge=300):
    h, w, c = img.shape
    if h<c or w<c:
        img = img.transpose(1,2,0)
        h, w, c = img.shape
    if h<=short_edge and w<=short_edge:
        return (w, h)
    if h>=w:
        dim = (int(w*long_edge/h), long_edge)
        return dim
    else:
        dim = (long_edge, int(h*long_edge/w))
        return dim

def get_new_dim_img(img, long_edge=300, down_sample='crop'):
    assert down_sample in ('crop', 'resize')
    h, w, c = img.shape
    if h<c or w<c:
        img = img.transpose(1,2,0)
        h, w, c = img.shape
    if h<=long_edge or w<=long_edge:
        return img
    
    if down_sample == 'resize':
        if h>=w:
            dim = (int(w*long_edge/h), long_edge)
            return cv2.resize(img, dim)
        else:
            dim = (long_edge, int(h*long_edge/w))
            return cv2.resize(img, dim)    
    elif down_sample == 'crop':
        h0 = random.randint(0, h-long_edge)
        w0 = random.randint(0, w-long_edge)
        img1 = img[h0:h0+long_edge, w0:w0+long_edge,:]
        return img1


def get_main_color(img, k=10, max_iter=100, save_color_map=False, save_path=None):
    # 转换数据维度
    img_ori_shape = img.shape
    img1 = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))
    img1 = img1[img1.sum(axis=1) != 0, :]

    img_shape = img1.shape

    # 获取图片色彩层数
    n_channels = img_shape[1]

    estimator = KMeans(n_clusters=k, max_iter=100,
                       init='k-means++', n_init=50)  # 构造聚类器
    estimator.fit(img1)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心

    colorLabels = list(estimator.labels_)
    colorInfo = {}
    for center_index in range(k):
        colorRatio = colorLabels.count(center_index)/len(colorLabels)
        colorInfo[colorRatio] = centroids[center_index]

    # 根据比例排序，从高至第低
    colorInfo = [colorInfo[k] for k in sorted(colorInfo.keys(), reverse=True)]

    if save_color_map:
        # 使用算法跑出的中心点，生成一个矩阵，为数据可视化做准备
        result = []
        result_width = 100
        result_height_per_center = 20
        for center_index in range(k):
            result.append(np.full((result_width * result_height_per_center,
                                   n_channels), colorInfo[center_index], dtype=int))
        result = np.array(result)
        result = result.reshape(
            (result_height_per_center * k, result_width, n_channels))

        # 保存图片
        io.imsave(os.path.splitext(save_path, result))
    return colorInfo
