import freicar_dataloader as loader
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision


########################################################################
# Demo test script for the freicar dataloader for bounding boxes
# Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
########################################################################

def GenBBOverlay(bbs, draw_image=None):
    bb_image = np.zeros((draw_image.shape[0], draw_image.shape[1], 3)).astype(np.uint8)
    overlay_img = (draw_image.copy()).astype(np.uint8)
    for car in bbs.values():
        for bb in car:
            if draw_image is not None:
                print(bb['x'].data.cpu().numpy()[0])
                cv2.rectangle(bb_image, (bb['x'].data.cpu().numpy()[0], bb['y'].data.cpu().numpy()[0]),
                              (bb['x'].data.cpu().numpy()[0] + bb['width'].data.cpu().numpy()[0],
                               bb['y'].data.cpu().numpy()[0] + bb['height'].data.cpu().numpy()[0]),
                              color=(0, 0, 255), thickness=2)

    overlay = cv2.addWeighted(bb_image, 0.3, overlay_img, 0.7, 0)
    return overlay


def TensorImage3ToCV(data):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    return cv


def gaussian_blur(image, blur):
    image = cv2.GaussianBlur(image,(5,5),blur)
    return image


import os
p = os.path.dirname(os.path.abspath(__file__))

static_data = loader.FreiCarDataset("./data",
                                    padding=(0, 0, 12, 12),
                                    split='training',
                                    load_real=False)

train_loader = torch.utils.data.DataLoader(static_data,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=1,
                                           pin_memory=False, drop_last=True)


for nr, (sample) in enumerate(train_loader):
    image = sample['rgb'][0]
    bbs = sample['bbs']
    gaussian = torchvision.transforms.GaussianBlur(5, (0.5, 5.0))
    flip = torchvision.transforms.RandomHorizontalFlip(p=1)
    print(image.size())
    print(bbs)
    cv_rgb = TensorImage3ToCV(image)
    overlay = GenBBOverlay(bbs, cv_rgb)
    cv2.imshow('BB Overlay', overlay)

    image_flip = flip.forward(image)
    image_flip = gaussian.forward(image_flip)
    image_flip = TensorImage3ToCV(image_flip)
    bbs_flip = bbs
    print(image.size()[-1])
    print(bbs_flip['freicar_3'][0]['x'])
    print(bbs_flip['freicar_3'][0]['width'])
    image_width = image.size()[-1]
    for key in bbs.keys():
        for bbs_individual in bbs[key]:
            bbs_individual['x'] = image_width - (bbs_individual['x'] + bbs_individual['width'])

    overlay_flip = GenBBOverlay(bbs_flip, image_flip)

    cv2.imshow('flip ', overlay_flip)
    #cv2.imshow("Flip ", TensorImage3ToCV())

    #tensor_image = TensorImage3ToCV(image)

    #cv2.imshow('RGB', cv_rgb)
    cv2.waitKey()