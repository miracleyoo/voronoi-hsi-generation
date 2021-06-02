import pandas as pd
import numpy as np
import utils
from tifffile import imwrite
import cv2
import os.path as op
from mlib import mio

class Synthesizer:
    def __init__(self, camera_function: pd.DataFrame, config = {
                 'start_wavelength':300,
                 'end_wavelength':3000,
                 'start_threshold':250,
                 'end_threshold':1000,
                 'ignore_limits':False}
                 ):

        
        print(f'config: {config}')
        # Camera function
        self.camera_function = camera_function
        self.band_num = len(self.camera_function.columns)

        # Weighted sum configurations
        self.config = config

        # Path to pickle serialized dataframe
        self.data_selection_pkl = ''

        # Files array
        self.files_arr = np.array([])

        # Image array
        self.img_arr = np.array([])

    def generate_random_files(self, n=25, selection_pickle=None):
        """
        Generates random array of size: size from data selection pickle.
        Sets the self.file_array to this array

        :argument n number of element we want to sample
        :arg selection_pickle path of the pickle object that contains a dataframe
        of the filepath we want to use. Should atleast have the column ['file_path']

        :returns: files_arr - np array containing random sampled files
        """
        self.files_arr = np.array([])

        selection_df = pd.DataFrame()

        # If selection_pickle is not None, use the path
        # passed in from the argument and set self.data_selection_pkl
        # to be the argument passed in

        if selection_pickle is not None:
            selection_df = pd.read_pickle(selection_pickle)
            if len(self.data_selection_pkl) == 0:
                self.data_selection_pkl = selection_pickle

        else:
            try:
                selection_df = pd.read_pickle(self.data_selection_pkl)
            except Exception as e:
                print(str(e))

        # Get n samples of the dataframe
        # and return the numpy array
        sampled_df = selection_df.sample(n)

        files_arr =  np.array(sampled_df['file_path'])
        self.files_arr = files_arr

        # print(f'Generated random samples: \n {self.files_arr}')

        return files_arr

    def save_img(self,filename,ext='.tif',img=[]):
        ext = op.splitext(filename)[1]

        if len(img) == 0:
            to_save = self.img_arr
        else:
            to_save = img

        print(f'Saving to {filename}...')
        if ext == '.tif':
            imwrite(filename, to_save, planarconfig='CONTIG')
        else:
            mio.dump(to_save, filename)


    def generate_img_from_file(self, dim: (int,int), filename: str):
        """
        Doesn't return anything but saves the file under the filename specified
        First we reshape self.files_arr and then we find the weighted sum of each element

        :return:
        Doesn't return anything but sets img_arr to be the processed img_arr file
        """
        # Reset img_arr
        self.img_arr = []

        # print('Generating...')
        # print(f'Reshaping...\nself.files_arr shape:  {self.files_arr.shape}')
        reshaped_file_array = np.reshape(self.files_arr, dim)
        # print(f'Reshaping Done\nself.files_arr shape:  {reshaped_file_array.shape}')

        for row in reshaped_file_array:
            new_row = [utils.get_weighted_sums_from_txt_file(x, self.camera_function, config=self.config)[1] for x in row]
            self.img_arr.append(new_row)

        self.img_arr = np.array(self.img_arr)

        print(f'Image array shape: {self.img_arr.shape}')
        self.save_img(filename)


    def sample_img(self, ratio:float, save = False, filename = '' ):
        if len(self.img_arr) == 0:
            raise ValueError('No image to be sampled')

        sampled_image = cv2.resize(np.array(self.img_arr),  # original image
                                   (0, 0),  # set fx and fy, not the final size
                                   fx=ratio,
                                   fy=ratio,
                                   interpolation=cv2.INTER_CUBIC)

        if save:
            self.save_img(filename,img=sampled_image)

        return sampled_image

    def imshow(self, bands=3):
        return
    
    def generate_voronoi(self, width, height, num_cells,num_materials, materials_array=[], sample_random=False, save_img=True, filename='vor.tiff'):
        if len(materials_array) == 0 and sample_random == False:
            raise Warning('sample_random has to be True if materials_array is empty, proceeding with random sampling materials')
        
        usage_materials_array = materials_array
        if sample_random:
            usage_materials_array = self.generate_random_files(num_materials, selection_pickle=self.data_selection_pkl)

        weighted_array = [utils.get_weighted_sums_from_txt_file(x, self.camera_function, config=self.config)[1] for x in usage_materials_array]

        self.img_arr = utils.voronoi(width, height, num_cells, weighted_array, self.band_num)
        
        if save_img:
            self.save_img(filename, img=self.img_arr)
        
        return self.img_arr

    
    def set_files_arr(self, files_arr):
        self.files_arr = files_arr

    def set_selection_pickle(self, selection_pickle):
        self.data_selection_pkl = selection_pickle
