from torch import preserve_format
from Synthesizer import Synthesizer
import pandas as pd
from tqdm import trange
import random
import os
import math
import pickle as pkl
from pathlib import Path
import utils
from mlib import mio

D = {
    1: 64,
    2: 32,
    4: 16,
    8: 8,
    16: 4,
    32: 2,
    64: 1
}


class DataGenerator:
    def __init__(self, n, camera_function: pd.DataFrame, dirs='generated_images', synth_config={}, band_num=None):
        self.n = n
        self.synth = Synthesizer(camera_function, config=synth_config, band_num=band_num)
        self.dirs = dirs
        self.selection_file = 'ref/success.pkl'

        print(f'Created initializor for {n} to {self.dirs}')

    def generate_data(self, im_size, template='test', start=0):
        for i in trange(start, self.n+start):
            try:
                self.synth.generate_random_files(
                    im_size[0]*im_size[1], selection_pickle=self.selection_file)
                self.synth.generate_img_from_file(
                    dim=im_size, filename=f'{self.dirs}/{template}_{i}.tiff')
            except ValueError:
                continue

    def generate_voronoi(self, im_size, template='voronoi', start=0, min_num_cells=10, max_num_cells=20, min_num_materials=5, max_num_materials=10):

        # Add selection data pickle
        self.synth.set_selection_pickle(self.selection_file)

        for i in trange(start, self.n+start):
            try:
                rand_num_cells = random.randrange(min_num_cells, max_num_cells)
                rand_num_materials = random.randrange(
                    min_num_materials, max_num_materials)
                fn = os.path.join(self.dir, f'{template}_{i}.tiff')
                self.synth.generate_voronoi(
                    im_size[0], im_size[1], rand_num_cells, rand_num_materials, filename=fn, sample_random=True)
            except ValueError:
                continue

    def generator_sampling(self, im_size, img_type='checkboard', sampling_times=1, template='', start=0, config={
                                                                                            'min_num_cells': 20,
                                                                                            'max_num_cells': 40,
                                                                                            'min_num_materials': 10,
                                                                                            'max_num_materials': 25,
                                                                                            'ext': 'npy',
                                                                                            'template_file': '',
                                                                                            'pre_color_files':[]
    }):
        """

        Desired file structure:

        root
        |
        |-- image0001
                |
                |-- 1x.npy (12x32x32)
                |-- 2x.npy (12x64x64)
                |-- 4x.npy (12x128x128)
                |--...
                |--64x.npy (12x1024x1024)
        |-- image 0002
                |
                |-- ...

        """
        # Add selection data pickle
        self.synth.set_selection_pickle(self.selection_file)
        materials_array_total = []

        img_folder = template
        if len(template) == 0:
            img_folder = img_type

        i = start

        try:
            # Get function parameters from config
            min_num_cells = config['min_num_cells']
            max_num_cells = config['max_num_cells']
            min_num_materials = config['min_num_materials']
            max_num_materials = config['max_num_materials']
            ext = config['ext']
            template_file = config['template_file']
            pre_color_files = config['pre_color_files']
        except KeyError as e:
            print(f'KeyError: {str(e)}')

        if template_file != '':
            print(template_file)
            temp_img = mio.load(template_file)
            temp_img = utils.get_new_dim_img(temp_img)
            materials_array_total = utils.get_main_color(temp_img, k=max_num_materials)
        elif pre_color_files != []:
            if isinstance(pre_color_files, list):
                for color_file in pre_color_files:
                    with open(color_file, 'rb') as f:
                        materials_array_total.extend(pkl.load(f))
            elif isinstance(pre_color_files, str):
                with open(pre_color_files, 'rb') as f:
                    materials_array_total = pkl.load(f)

        while i < self.n + start:

            value_err = False

            if img_type == 'checkboard':
                # TODO: Complete here, just for fun
                pass
            elif img_type == 'voronoi':

                # Create full path here
                full_path = os.path.join(self.dirs, f'{img_folder}_{i}')
                Path(full_path).mkdir(parents=True,
                                      exist_ok=True)  # python 3.5 above

                # Use random to generate the parameter passed into the functions
                rand_num_cells = random.randrange(min_num_cells, max_num_cells)
                rand_num_materials = random.randrange(
                    min_num_materials, max_num_materials)

                for s in range(sampling_times+1):
                    cur_rate = int(math.pow(2, s))
                    print('cur_rate:', cur_rate, 'D:', D)
                    fn = os.path.join(full_path, f'{D[cur_rate]}x.{ext}')
                    if cur_rate == 1:
                        try:
                            if materials_array_total == []:
                                self.synth.generate_voronoi(
                                    im_size[0], im_size[1], rand_num_cells, rand_num_materials, filename=fn, sample_random=True)
                            else:
                                materials_array = random.sample(materials_array_total, rand_num_materials)
                                self.synth.generate_voronoi(
                                    im_size[0], im_size[1], rand_num_cells, rand_num_materials, materials_array=materials_array, filename=fn, sample_random=False)
                        except ValueError as e:
                            print(f'ValueERR: {str(e)} Retrying...')
                            value_err = True
                            break
                    else:
                        sample_ratio = 1.0/cur_rate
                        self.synth.sample_img(
                            sample_ratio, save=True, filename=fn)

            else:
                raise ValueError(
                    "Img type not recognized, value available: 'chessboard', 'voronoi'")

            if not value_err:
                i += 1


## Sentinel-Simulation
# scf = {
#     'step':5,
#     'start_wavelength':300,
#     'end_wavelength':3000,
#     'start_threshold':250,
#     'end_threshold':1000,
#     'ignore_limits':True,
#     'template_file':''
# }

# camera_df = pd.read_pickle('ref/sen_norm_df.pkl') #ideal_norm_df
# NUM_IMG = 10
# DIRS = 'generated_images'

## Ideal-Simulation
# scf = {
#     'step': 1,
#     'start_wavelength': 388,
#     'end_wavelength': 1013,
#     'start_threshold': 250,
#     'end_threshold': 1000,
#     'ignore_limits': True,
#     'template_file':''
# }

# camera_df = pd.read_pickle('ref/ideal_norm_df.pkl')
# NUM_IMG = 10
# DIRS = 'generated_images'


## Sentinel-Template-Simulation
# scf = {
#     'step': 5,
#     'start_wavelength': 300,
#     'end_wavelength': 3000,
#     'start_threshold': 250,
#     'end_threshold': 1000,
#     'ignore_limits': True
# }

# camera_df = pd.read_pickle('ref/sen_norm_df.pkl')
# NUM_IMG = 100
# DIRS = r'L:\Satellite\synthetic_sentinel_real'


# dg = DataGenerator(NUM_IMG, camera_df, dirs=DIRS, synth_config=scf, band_num=12)

# dg.generator_sampling((1024, 1024), img_type='voronoi', template='voronoi1024', sampling_times=6, start=1, config={
#     'min_num_cells': 80,
#     'max_num_cells': 100,
#     'min_num_materials': 10,
#     'max_num_materials': 25,
#     'ext': 'pkl',
#     'template_file': r'L:\Satellite\paired_dl_data\chand1\sentinel\roi.npy'
# })


scf = {
    'step': 5,
    'start_wavelength': 300,
    'end_wavelength': 3000,
    'start_threshold': 250,
    'end_threshold': 1000,
    'ignore_limits': True
}

camera_df = pd.read_pickle('ref/sen_norm_df.pkl')
NUM_IMG = 2000
DIRS = r'L:\Satellite\synthetic_sentinel_mix_v2'

os.makedirs(DIRS, exist_ok=True)
dg = DataGenerator(NUM_IMG, camera_df, dirs=DIRS, synth_config=scf, band_num=31)

# dg.generator_sampling((1024, 1024), img_type='voronoi', template='voronoi1024', sampling_times=6, start=1, config={
#     'min_num_cells': 80,
#     'max_num_cells': 100,
#     'min_num_materials': 10,
#     'max_num_materials': 25,
#     'ext': 'pkl',
#     'template_file': '',
#     'pre_color_files': ['ref/ntire_colors_10per.pkl', 'ref/icvl_colors_10per.pkl']
# })

dg.generator_sampling((1024, 1024), img_type='voronoi', template='voronoi1024', sampling_times=6, start=1, config={
    'min_num_cells': 50,
    'max_num_cells': 250,
    'min_num_materials': 10,
    'max_num_materials': 100,
    'ext': 'pkl',
    'template_file': '',
    'pre_color_files': ['ref/ntire_colors_10per.pkl', 'ref/icvl_colors_10per.pkl']
})