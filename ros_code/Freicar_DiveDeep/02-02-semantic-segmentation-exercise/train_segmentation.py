import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import cv2
import numpy as np
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import fast_scnn_model
from dataset_helper import freicar_segreg_dataloader
from dataset_helper import color_coder
import matplotlib.pyplot as plt

#################################################################
# AUTHOR: Johan Vertens (vertensj@informatik.uni-freiburg.de)
# DESCRIPTION: Training script for FreiCAR semantic segmentation
##################################################################

def visJetColorCoding(name, img):
    img = img.cpu().byte().data.numpy().squeeze()
    color_img = np.zeros(img.shape, dtype=(img.astype(np.uint8)).dtype)
    cv2.normalize(img, color_img, 0, 255, cv2.NORM_MINMAX)
    color_img = color_img.astype(np.uint8)
    color_img = cv2.applyColorMap(color_img, cv2.COLORMAP_JET, color_img)
    cv2.imshow(name, color_img)


def visImage3Chan(data, name):
    print(data.shape)
    cv = np.transpose(data.squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)


parser = argparse.ArgumentParser(description='Segmentation and Regression Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, help="Start at epoch X")
parser.add_argument('--batch_size', default=10, type=int, help="Batch size for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lane_regression', '-l', default=False, type=bool, help='Lane regression flag')
best_iou = 0
args = None
lane_loss_regularizer = 5
writer = SummaryWriter()


def main():
    global args, best_iou
    args = parser.parse_args()

    # Create Fast SCNN model...
    model = fast_scnn_model.Fast_SCNN(3, 4)
    model = model.cuda()

    # Number of max epochs, TODO: Change it to a reasonable number!
    num_epochs = 20

    optimizer = Adam(model.parameters(), 5e-3)
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epochs)), 0.9)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume)['state_dict'])
            args.start_epoch = 0
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0

    # Data loading code
    load_real_images = False
    train_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                       split='training', load_real=load_real_images)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=False, drop_last=True)
    criterion = CrossEntropyLoss()
    l1loss = L1Loss()

    # If --evaluate is passed from the command line --> evaluate
    if args.evaluate:
        eval_dataset = freicar_segreg_dataloader.FreiCarLoader("../data/", padding=(0, 0, 12, 12),
                                                                split='validation', load_real=load_real_images)

        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                   pin_memory=False, drop_last=False)

        eval(eval_loader, model, criterion, l1loss)

    iou_0 = []
    iou_1 = []
    iou_2 = []
    iou_3 = []
    mious = []
    losses = []
    lane_reg_losses = []

    iou_0_val = []
    iou_1_val = []
    iou_2_val = []
    iou_3_val = []
    mious_val = []
    val_losses = []
    val_lane_reg_losses = []

    best_loss = 9999

    use_lane_regression = False
    if args.lane_regression:
        use_lane_regression = True


    for epoch in range(args.start_epoch, num_epochs):
        # train for one epoch
        loss, ious = train(train_loader, model, optimizer, scheduler, epoch, criterion, l1loss, use_lane_regression)

        losses.append(loss[0])
        lane_reg_losses.append(loss[1])
        iou_0.append(ious[0])
        iou_1.append(ious[1])
        iou_2.append(ious[2])
        iou_3.append(ious[3])
        miou = (ious[0] + ious[1] + ious[2] + ious[3]) / 4
        mious.append(miou)


        #print(np.mean(np.array(iou_0)), np.mean(np.array(iou_1)), np.mean(np.array(iou_2)), np.mean(np.array(iou_3)))
        if args.evaluate:
            val_loss, ious_val = eval(eval_loader, model, criterion, l1loss, use_lane_regression)
            val_losses.append(val_loss[0])
            val_lane_reg_losses.append(val_loss[1])
            iou_0_val.append(ious_val[0])
            iou_1_val.append(ious_val[1])
            iou_2_val.append(ious_val[2])
            iou_3_val.append(ious_val[3])
            miou_val = (ious_val[0] + ious_val[1] + ious_val[2] + ious_val[3]) / 4
            mious_val.append(miou_val)

            if best_loss > val_loss[0]:
                best_loss = val_loss[0]
                # remember best iou and save checkpoint
                print("Writing best model ", epoch + 1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_iou': best_iou,
                    'optimizer': optimizer.state_dict(),
                }, True)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_iou': best_iou,
                    'optimizer': optimizer.state_dict(),
                }, False, "recent_model_"+str(epoch+1)+".pth.tar")


            print("Epoch: ", str(epoch) + " -> Training loss and miou: ", loss[0],
                  miou, "  -> Validation loss and miou: ", val_loss[0],
                  miou_val)
            if use_lane_regression:
                print("Training lr loss: ", loss[1], " -> validation lr loss: ", val_loss[1])

        else:
            print("Epoch: ", str(epoch) + " -> Training loss and miou: ", loss[0], miou)
            if use_lane_regression:
                print("Training lr loss: ", loss[1])


        plt1 = plt.figure()
        plt.plot(np.arange(len(mious)), mious, label="mIoU")
        plt.plot(np.arange(len(mious)), iou_0, label="iou_0")
        plt.plot(np.arange(len(mious)), iou_1, label="iou_1")
        plt.plot(np.arange(len(mious)), iou_2, label="iou_2")
        plt.plot(np.arange(len(mious)), iou_3, label="iou_3")
        plt.xlabel("epoch")
        plt.ylabel("iou")
        plt.title("Training iou")
        plt.legend()
        plt.grid()
        if use_lane_regression:
            plt1.savefig("lr_train_iou")
        else:
            plt1.savefig("train_iou")

        if args.evaluate:
            plt2 = plt.figure()
            plt.plot(np.arange(len(mious_val)), mious_val, label="mIoU")
            plt.plot(np.arange(len(mious_val)), iou_0_val, label="iou_0")
            plt.plot(np.arange(len(mious_val)), iou_1_val, label="iou_1")
            plt.plot(np.arange(len(mious_val)), iou_2_val, label="iou_2")
            plt.plot(np.arange(len(mious_val)), iou_3_val, label="iou_3")
            plt.xlabel("epoch")
            plt.ylabel("iou")
            plt.title("Validation iou")
            plt.legend()
            plt.grid()
            if use_lane_regression:
                plt2.savefig("lr_validation_iou")
            else:
                plt2.savefig("validation_iou")

        plt3 = plt.figure()
        plt.plot(np.arange(len(losses)), losses, label="training loss")
        plt.title("Training loss")
        if args.evaluate:
            plt.plot(np.arange(len(losses)), val_losses, label="validation loss")
            plt.title("Training and validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()
        plt.grid()
        if use_lane_regression:
            plt3.savefig("lr_loss")
        else:
            plt3.savefig("loss")

        if use_lane_regression:
            plt4 = plt.figure()
            plt.plot(np.arange(len(lane_reg_losses)), lane_reg_losses, label="training lr loss")
            plt.title("Training lane regression loss")
            if args.evaluate:
                plt.plot(np.arange(len(lane_reg_losses)), val_lane_reg_losses, label="validation lr loss")
                plt.title("Training and validation lane regression loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.grid()
            plt4.savefig("lane_regression_loss")


        plt.close('all')

        print("Training miou, iou_0, iou_1, iou_2, iou_3: ", miou, ious[0], ious[1], ious[2], ious[3])
        if args.evaluate:
            print("Validation miou, iou_0, iou_1, iou_2, iou_3: ", miou_val, ious_val[0], ious_val[1], ious_val[2], ious_val[3])

        if epoch % 10 == 0:
            print("training losses: ")
            print(losses)
            if use_lane_regression:
                print(lane_reg_losses)
            print("mious: ")
            print(mious)
            print("ious")
            print(iou_0)
            print(iou_1)
            print(iou_2)
            print(iou_3)

            if args.evaluate:
                print("val losses: ")
                print(val_losses)
                if use_lane_regression:
                    print(val_lane_reg_losses)
                print("mious_val: ")
                print(mious_val)
                print("ious: ")
                print(iou_0_val)
                print(iou_1_val)
                print(iou_2_val)
                print(iou_3_val)


def train(train_loader, model, optimizer, scheduler, epoch, criterion, l1loss, use_lane_regression=False):
    global lane_loss_regularizer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    iou_0 = AverageMeter()
    iou_1 = AverageMeter()
    iou_2 = AverageMeter()
    iou_3 = AverageMeter()
    lane_reg_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    #train_loader = tqdm(train_loader)
    for i, (sample) in enumerate(train_loader):

        data_time.update(time.time() - end)

        image = sample['rgb'].cuda().float()
        lane_reg = sample['reg'].cuda().float()
        seg_ids = sample['seg'].cuda()

        ######################################
        # TODO: Implement me! Train Loop
        optimizer.zero_grad()
        sg_output, lane_output = model(image)
        seg_ids = torch.squeeze(seg_ids, dim=1)
        if use_lane_regression:
            class_loss = criterion(sg_output, seg_ids.long())
            lane_reg_loss = l1loss(lane_output, lane_reg)*lane_loss_regularizer
            loss =  class_loss + lane_reg_loss
            lane_reg_losses.update(lane_reg_loss.item())
        else:
            loss = criterion(sg_output, seg_ids.long())
        loss.backward()
        optimizer.step()

        ious = mIoU(sg_output, seg_ids)
        # In some images a class of pixels might not exists and union of them might be zero. In such cases we get -1 as
        # the iou and we ignore those values for calculation
        if ious[0] != -1:
            iou_0.update(ious[0])
        if ious[1] != -1:
            iou_1.update(ious[1])
        if ious[2] != -1:
            iou_2.update(ious[2])
        if ious[3] != -1:
            iou_3.update(ious[3])

        ######################################
        #log values to tensorboard
        writer.add_scalar("Loss/train", loss, epoch)
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0) )
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


    scheduler.step(epoch)
    return ((losses.avg, lane_reg_losses.avg), (iou_0.avg, iou_1.avg, iou_2.avg, iou_3.avg))


def TensorImage3ToCV(data):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    return cv


def eval(data_loader, model, criterion, l1loss, use_lane_regression=False):
    global lane_loss_regularizer
    model.eval()
    color_conv = color_coder.ColorCoder()
    losses = AverageMeter()
    iou_0 = AverageMeter()
    iou_1 = AverageMeter()
    iou_2 = AverageMeter()
    iou_3 = AverageMeter()
    lane_reg_losses = AverageMeter()

    with torch.no_grad():
        for i, (sample) in enumerate(data_loader):

            image = sample['rgb'].cuda().float()
            lane_reg = sample['reg'].cuda().float()
            seg_ids = sample['seg'].cuda()

            #print(image.size())
            seg_pred, lane_pred = model(image)
            ######################################
            # TODO: Implement me!
            # You should calculate the IoU every N
            # epochs for both the training
            # and the evaluation set.
            # For segmentation color coding
            # see: color_coder.py

            seg_ids = torch.squeeze(seg_ids, dim=1)
            if use_lane_regression:
                class_loss = criterion(seg_pred, seg_ids.long())
                lane_reg_loss = l1loss(lane_pred, lane_reg)*lane_loss_regularizer
                loss = class_loss + lane_reg_loss
                lane_reg_losses.update(lane_reg_loss.item())
            else:
                loss = criterion(seg_pred, seg_ids.long())

            ious = mIoU(seg_pred, seg_ids)
            cv_rgb = TensorImage3ToCV((sample['rgb'] * 255.).byte())
            #cv2.imshow('RGB', cv_rgb)
            # pred_rgb_colorcoded = color_conv.color_code_labels(seg_pred)
            # seg_ids_rgb_colorcoded = color_conv.color_code_labels(seg_ids, argmax=False)
            # 
            # visJetColorCoding('Reg GT', lane_reg)
            # visJetColorCoding('Reg PRED', lane_pred)
            # 
            # print('Seg_ID', pred_rgb_colorcoded.shape, 'Pred', seg_ids_rgb_colorcoded.shape)
            # cv2.imshow('Segmentation Predection', pred_rgb_colorcoded)
            # 
            # cv2.imshow('Segmentation Ground Truth', seg_ids_rgb_colorcoded)
            # cv2.waitKey()

            #In some images a class of pixels might not exists and union of them might be zero. In such cases we get -1 as
            # the iou and we ignore those values for calculation
            if ious[0] != -1:
                iou_0.update(ious[0])
            if ious[1] != -1:
                iou_1.update(ious[1])
            if ious[2] != -1:
                iou_2.update(ious[2])
            if ious[3] != -1:
                iou_3.update(ious[3])
            losses.update(loss.item())

    return ((losses.avg, lane_reg_losses.avg), (iou_0.avg, iou_1.avg, iou_2.avg, iou_3.avg))
    ######################################



def mIoU(logits, gt):
    ious = []
    pred = logits.argmax(dim=1)
    for i in range(4):

        intersection = ((pred == i) & (gt == i)).sum().float()
        union = ((pred == i) | (gt == i)).sum().float()

        if union.item() != 0:
            ious.append(intersection.item() / union.item())
        else:
            ious.append(-1)
    return ious

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
