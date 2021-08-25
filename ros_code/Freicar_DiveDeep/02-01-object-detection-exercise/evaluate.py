"""
COCO-Style Evaluations
"""
import argparse
import torch
import yaml
from tqdm import tqdm
from model.efficientdet.backbone import EfficientDetBackbone
from model.efficientdet.utils import BBoxTransform, ClipBoxes
from utils import postprocess, boolean_string
from dataloader.freicar_dataloader import FreiCarDataset
from model.efficientdet.dataset import collater
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
########################################################################
# Object Detection model evaluation script
# Modified by: Jannik Zuern (zuern@informatik.uni-freiburg.de)
########################################################################
def metrics(predections: np.ndarray, labels: np.ndarray, iou_threshold: float):
    # Sum is used to identify ground truth with zero bounding box.
    sum = np.sum(labels)
    ious = []
    FP = 0
    TP = 0
    FN = 0
    # False positive case
    if sum == -5:
        FP = 1 * len(predections)
    # False Negative case
    elif len(predections) == 0:
        FN = len(labels)
        ious = [0] * len(labels)
    else:
        if len(labels) < len(predections):
            #These are unnecessary extra boxes
            FP = len(predections) - len(labels)
            ious = [0] * len(labels)
            for i in range(len(labels)):
                compare_iou = []
                # we are calculating iou of each ground truth w.r.t every predicted bounding box and selecting the best one
                # We using this logic to make sure if the prediction order is not inaccordence with ground truth order in case of multiple objects
                for j in range(len(predections)):
                    intersection = (min(predections[j][2], labels[i][2]) - max(predections[j][0], labels[i][0])) * \
                                   (min(predections[j][3], labels[i][3]) - max(predections[j][1], labels[i][1]))
                    union = ((predections[j][2] - predections[j][0]) * (predections[j][3] - predections[j][1])) + (
                            (labels[i][2] - labels[i][0]) *
                            (labels[i][3] - labels[i][1])) - intersection
                    compare_iou.append(intersection / union)
                ious[i] = np.max(compare_iou)
        else:
            ious = [0] * len(labels)
            for j in range(len(predections)):
                compare_iou = []
                # we are calculating iou of each ground truth w.r.t every predicted bounding box and selecting the best one
                # We using this logic to make sure if the prediction order is not inaccordence with ground truth order in case of multiple objects
                for i in range(len(labels)):
                    intersection = (min(predections[j][2], labels[i][2]) - max(predections[j][0], labels[i][0])) * \
                                   (min(predections[j][3], labels[i][3]) - max(predections[j][1], labels[i][1]))
                    union = ((predections[j][2] - predections[j][0]) * (predections[j][3] - predections[j][1])) + (
                            (labels[i][2] - labels[i][0]) *
                            (labels[i][3] - labels[i][1])) - intersection
                    compare_iou.append(intersection / union)
                ious[j] = np.max(compare_iou)
        for iou in ious:
            #There is a box but we are discarding because of threshold. Hence we add it to FN
            if iou < iou_threshold:
                FN += 1
            else:
                TP += 1
    return {
        "ious": ious,
        "tp": TP,
        "fp": FP,
        "fn": FN
    }
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='freicar-detection', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
args = ap.parse_args()
compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
project_name = args.project
weights_path = args.weights
params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']
threshold = 0.2
iou_threshold = 0.2
if __name__ == '__main__':
    '''
    Note: 
    When calling the model forward function on an image, the model returns
    features, regression, classification and anchors.
    In order to obtain the final bounding boxes from these predictions, they need to be postprocessed
    (this performs score-filtering and non-maximum suppression)
    Thus, you should call
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    preds = postprocess(imgs, anchors, regression, classification, regressBoxes, clipBoxes, threshold, nms_threshold)                  
    preds = preds[0]
    Now, the scores, class_indices and bounding boxes are saved as fields in the preds dict and can be used for subsequent evaluation.
    '''
    set_name = 'validation'
    freicar_dataset = FreiCarDataset(data_dir="./dataloader/data/",
                                     padding=(0, 0, 12, 12),
                                     split=set_name,
                                     load_real=False)
    val_params = {'batch_size': 1,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': 1}
    freicar_generator = DataLoader(freicar_dataset, **val_params)
    # instantiate model
    model = EfficientDetBackbone(compound_coef=compound_coef,
                                     num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']),
                                     scales=eval(params['anchors_scales']))
    # load model weights file from disk
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    ##########################################
    # TODO: implement me!
    precision_list = []
    recall_list = []
    miou_list = []
    for j in range(0, 11):
        threshold = j * 0.1
        print("===================================")
        print(threshold)
        model.eval()
        ious = []
        progress_bar = tqdm(freicar_generator)
        # when iou threshold = 1 false -ve and true +ve will be equal to zero we get divide by zero error so we are setting TP =1 as default.
        TP = 1
        FN = 0
        FP = 0
        for i, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    inputs = data['img'].float()
                    labels = data['annot']
                    features, regression, classification, anchors = model(inputs)
                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()
                    preds = postprocess(inputs, anchors, regression, classification, regressBoxes, clipBoxes, threshold,
                                        nms_threshold)
                    #print(labels)
                    preds = preds[0]
                    #print(preds)
                    labels = torch.squeeze(labels)
                    data = metrics(preds['rois'].reshape(-1, 4), labels.cpu().detach().numpy().reshape(-1, 5), iou_threshold)
                    #print(data)
                    ious += data['ious']
                    TP += data['tp']
                    FP += data['fp']
                    FN += data['fn']
                    #TP, TN, FP, FN = confusion_matrix(preds['rois'].reshape(-1,4), labels.cpu().detach().numpy().reshape(-1, 5), miou)
                except Exception as e:
                    print(e)
        miou = np.mean(ious)
        print("miou : ", miou)
        precision = TP / (TP  + FP)
        print("True positive: ", TP)
        print("FalsePositive: ", FP)
        print("False Negative: ", FN)
        print("Precision: ", precision)
        recall = TP / (TP + FN)
        print("recall : ", recall)
        precision_list.append(precision)
        recall_list.append(recall)
        miou_list.append(miou)
    ##########################################
precision_list = [precision_list[i] for i in np.argsort(recall_list)[::-1]]
recall_list = [recall_list[i] for i in np.argsort(recall_list)[::-1]]
plt.plot([recall_list[i] for i in np.argsort(precision_list)[::-1]], [precision_list[i] for i in np.argsort(precision_list)[::-1]])
plt.title("PR curve plot")
plt.xlabel("recall")
plt.ylabel("precision")
plt.grid()
plt.savefig("precision recall curve.png")
print("miou:")
print(miou)
print("recall:")
print(recall_list)
print("precision")
print(precision_list)
aoc = 0
for i in range(len(recall_list)-1):
    aoc += ((recall_list[i] - recall_list[i+1]) * precision_list[i])
print("map: ", aoc)