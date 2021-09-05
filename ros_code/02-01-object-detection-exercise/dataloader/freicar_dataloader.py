import numpy as np
import torch.utils.data as data
import os
import os.path
import torch
import yaml
from pathlib import Path
import requests
from model.efficientdet.loss import FocalLoss
from zipfile import ZipFile
import sys
import cv2
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import *


########################################################################
# Demo freicar dataloader for bounding boxes
# Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
########################################################################

def flip(image, bbs):
    image = torch.flip(image, [2])
    image_width = image.size()[-1]
    for key in bbs.keys():
        for bbs_individual in bbs[key]:
            bbs_individual['x'] = image_width - (bbs_individual['x'] + bbs_individual['width'])
    return image, bbs


def data_transform(img, bbs):
    # We have implemented 2 dataaugmentation techniques horizontal flip and brightness change of the image
    # with probability 0.46 brightnes change, 0.02 horizontal flip and 0.02 both brightness and horizontal flip rest of the 50% images without any data augmentation.
    image = img[0]
    choice = np.random.choice([0, 1, 2, 3], p=[0.5, 0.46, 0.02, 0.02])
    if choice == 3:
        image, bbs = flip(image, bbs)
    elif choice == 2:
        image, bbs = flip(image, bbs)
        ratio = np.random.uniform(0, 3, 1)
        image = (ratio[0] * image).clamp(0, 255).to(image.dtype)
    elif choice == 1:
        ratio = np.random.uniform(0, 3, 1)
        image = (ratio[0] * image).clamp(0, 255).to(image.dtype)

    '''image = img[0]
    choice = np.random.choice([0, 1, 2, 3, 4], p=[0.5, 0.15, 0.1, 0.2, 0.05])
    if choice == 4:
        flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        image = flip.forward(image)

        image_width = image.size()[-1]
        for key in bbs.keys():
            for bbs_individual in bbs[key]:
                bbs_individual['x'] = image_width - (bbs_individual['x'] + bbs_individual['width'])
    elif choice == 3:
        gaussian = torchvision.transforms.GaussianBlur((31, 31), (0.1, 4.0))
        image = gaussian.forward(image)
        color_jitter = ColorJitter((0.1, 0.5), (.1, 0.5), (.1, 0.5), (0.1, 0.5))
        image = color_jitter.forward(image)
    elif choice == 2:
        color_jitter = ColorJitter((0.1, 0.5), (.1, 0.5), (.1, 0.5), (0.1, 0.5))
        image = color_jitter.forward(image)
    elif choice == 1:
        gaussian = torchvision.transforms.GaussianBlur((31, 31), (0.1, 4.0))
        image = gaussian.forward(image)'''

    return [image], bbs

def searchForFiles(name, search_path):
    paths = []
    for path in Path(search_path).rglob(name):
        paths.append(path)
    return paths


def loadRGB(paths, size=None, pad_info=None):
    rgb_im = [cv2.imread(p) for p in paths]
    if size is not None:
        rgb_im = [cv2.resize(im, size) for im in rgb_im]
    rgb_im = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_im]
    rgb_im = [(torch.from_numpy(im)).permute(2, 0, 1) for im in rgb_im]

    if pad_info is not None:
        rgb_im = [torch.nn.functional.pad(d, pad_info, 'constant', 0) for d in rgb_im]

    return rgb_im


def generatePaths(fnames, type):
    npaths = []
    for fn in fnames:
        npaths.append(fn.replace('synth', type))
    return npaths


class FreiCarDataset(data.Dataset):
    def downloadData(self, data_dir):

        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        link = 'http://aisdatasets.informatik.uni-freiburg.de/freicar/detection_data.zip'
        file_name = data_dir + "/detection_data.zip"

        with open(file_name, "wb") as f:
            print("Downloading %s" % file_name)
            response = requests.get(link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()

        print('Unzipping data...')
        with ZipFile(data_dir + '/detection_data.zip', 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(data_dir)

        print('Removing zipped file ....')
        os.remove(data_dir + '/detection_data.zip')

    def filterEvalFiles(self, all_paths, eval_paths):
        out = []
        for p in all_paths:
            if p.split('/')[-1] not in eval_paths:
                out.append(p)
        return out

    def filterTrainFiles(self, all_paths, eval_paths):
        out = []
        for p in all_paths:
            if p.split('/')[-1] in eval_paths:
                out.append(p)
        return out

    def __init__(self, data_dir, padding, split='training', load_real=False, transform_data=False):
        db_path = data_dir + '/detection_data/'

        if not os.path.exists(db_path):
            self.downloadData(data_dir)

        self.transform_data = transform_data
        self.db_path = db_path
        self.load_real = load_real

        eval_name_file = open(db_path + 'eval_names.txt', 'r')
        self.eval_files = eval_name_file.readlines()
        self.eval_files = [f.rstrip() for f in self.eval_files]

        self.eval_files = [f.split('/')[-1] for f in self.eval_files]

        self.rgb_path_files = sorted(searchForFiles('*.png', db_path + '/synth/'),
                                     key=lambda y: float(y.name.split('.')[0].replace('_', '.')))
        self.rgb_path_files = [str(p) for p in self.rgb_path_files]

        if split == 'training':
            self.rgb_path_files = self.filterEvalFiles(self.rgb_path_files, self.eval_files)
        elif split == 'validation':
            self.rgb_path_files = self.filterTrainFiles(self.rgb_path_files, self.eval_files)
        else:
            print('Split: %s not known...' % split)

        if self.load_real:
            self.real_rgb_paths = generatePaths(self.rgb_path_files, 'real')
            self.rgb_path_files.extend(self.real_rgb_paths)

        self.padding = padding
        self.info_file = searchForFiles('info.yaml', db_path + '/info/')[0]

        assert (self.rgb_path_files.__len__() > 0)

        with open(str(self.info_file)) as file:
            self.info = yaml.load(file, Loader=yaml.FullLoader)

        self.length = len(self.rgb_path_files)

        self.filenames = [os.path.basename(p).replace('.png', '') for p in self.rgb_path_files]

        print('Size of FreiCAR detection dataset, {} split: {} images'.format(split, self.length))

    def __getitem__(self, index):

        ##########################################
        # TODO: implement me!
        if self.load_real:
            image = loadRGB([(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                            self.db_path, "real", self.filenames[index] + ".png")))], pad_info=self.padding)
        else:
            image = loadRGB([(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),  "..",
                                                            self.db_path, "synth", self.filenames[index] + ".png")))], pad_info=self.padding)
        bbs = self.info[self.filenames[index]]

        for key in bbs.keys():
            for value    in bbs[key]:
                value['y'] = value['y'] + 12

        if self.transform_data:
            image, bbs = data_transform(image, bbs)

        item = {"rgb": image, "bbs": bbs}
        #print("image name: 1603450738_56504683")
        #print(bbs)
        return item
        ##########################################

        '''
        Note 1: 
        The subsequently called collater (defined in model/efficientdet/dataloader.py) expects a dict as a return value
        from this function. The dict should have to following form:
        out_dict['rgb']: rgb image
        out_dict['bbs']: bounding boxes for this image as saved in self.info
        
        Note 2: 
        We are padding the image with (padding_left, padding_right, padding_top, padding_bottom) = (0, 0, 12, 12)
        when loading the files from disk in order to resize them to (640 x 384), because EfficientDet-D0 expects
        image dimensions to be a multiple of 32. This is NOT reflected in the bounding box coordinates in the downloaded dataset.
        Thus, make sure that you add the correct offset to the bounding box coordinates for the padded images in this function.
        '''


    def __len__(self):
        return len(self.rgb_path_files)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
