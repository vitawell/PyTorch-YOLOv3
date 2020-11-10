from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable

    OpenCV 会将图像载入为 numpy 数组，颜色通道的顺序为 BGR。PyTorch 的图像输入格式是（batch x 通道 x 高度 x 宽度），其通道顺序为 RGB。
    因此，用 prep_image 来将 numpy 数组转换成 PyTorch 的输入格式。
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


# 转化为输出图片
def write(x, img):
    """
    在边界框的左上角创建一个填充后的矩形，并且写入在该框位置检测到的目标的类别。
    cv2.rectangle 函数的 -1 参数用于创建填充的矩形。

    x中的信息是图像索引、4个角坐标、目标置信度得分conf、最大置信类得分cls_conf、该类的索引cls_pred
    """

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())

    cls = int(x[-1])

    conf = float(x[5])
    cls_conf = float(x[6])
    cls_pred = float(x[7])

    label = "{0}".format(classes[cls])  # 类别文字，什么格式? 

    #输出类别的置信度需大于阈值
    if cls_conf > cls_thres :
        # bbox_colors = random.sample(colors, len(classes[cls]))
        # color = random.choice(colors)
        color = (43,137,193)

        cv2.rectangle(img, c1, c2, color, 1) #因为cv与python版本问题？参数需要为int整数

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)

        # putText 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    
    return img


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    # 输入视频的名称
    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="video.avi", type=str)
    # 输出视频的名称
    parser.add_argument("--output", dest='output', help=
    "Video to output",
                        default="video_output.avi", type=str)

    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="config/yolov3-custom.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)

    # 添加自己训练的模型
    parser.add_argument("--weights_path", dest='weights_path', type=str, default="checkpoints/yolov3_ckpt_1_56.pth",
                        help="path to weights file")
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    # 增加一个类别置信度阈值
    parser.add_argument("--cls_thres", type=float, default=0.7,help="object class confidence threshold")

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    cls_thres = args.cls_thres
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 4

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    if args.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(args.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(args.weights_path))

    # model.load_weights(args.weightsfile)
    model.eval()  # Set in evaluation mode

    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    # model(get_test_input(inp_dim, CUDA), CUDA)
    #
    # model.eval()

    videofile = args.video

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    #获取视频fps和宽高
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    #准备写视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # (*"MJPG")的"MJPG"需改成"mp4v"
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, size, True)

    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            # 转换图像
            img, orig_im, dim = prep_image(frame, inp_dim)
            # im_dim包含原始图像的维度
            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)   # 通过模型处理图片
                output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                # cv2.imshow("frame", orig_im)
                writer.write(orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            """
            在绘制边界框之前，输出tensor中包含的预测是对填充图像的预测，而不是对原始图像的预测。
            仅仅将它们重新缩放到输入图像的维数在这里是行不通的。将边界框的坐标转换到相对于包含原始图像的填充图像上的边界。
            """
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            # 有些边界框可能超出了图像边缘，我们要将其限制在图片范围内。
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            classes = load_classes('data/custom/classes.names')

            # colors = pkl.load(open("pallete", "rb"))
            # 随机颜色
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")


            # orig_im为输出图片
            list(map(lambda x: write(x, orig_im), output))

            # 一帧一帧显示，写视频时也得保留这段
            # 在colab上需注释这一句，负责会提示connot connect to X sever（云端没有图形界面，所以报错）
            # cv2.imshow("frame", orig_im)  #直接保存，不显示
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

            # 写入当前帧
            writer.write(orig_im)

        else:
            break

    # 释放文件指针
    print("[INFO] cleaning up...")
    writer.release()
    cap.release()

