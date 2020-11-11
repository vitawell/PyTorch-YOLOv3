from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 训练的轮次
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")  
    # 每次放进模型的批次
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")  
    # 累积多少部的梯度
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")  
    # yolo 配置文件路径
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")  
    # 也是配置文件，配置类别数、训练和测试集路径、类别名称文件路径等
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")  
    # 预训练模型的权重路径，最开始可以使用 weights/yolov3.weights 权重进行训练，也可以使用checkpoints
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")  

    # 生成数据是 cpu 的线程数
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")  
    # 输入数据的尺寸，此值必须是 32 的整数倍
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")  
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")  
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")  # 用来记录训练的日志

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是不是有 GPU 可以使用

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)  # parse_data_config 在 utils/parse_config.py 文件中，类型是字典
    train_path = data_config["train"]  # 训练集的路径
    valid_path = data_config["valid"]  # 验证集的路径
    class_names = load_classes(data_config["names"])  #  load_classes 在  utils/utils.py 文件中，类型是列表，其中每个值是一个类别，如阿猫啊狗

    # Initiate model
    model = Darknet(opt.model_def).to(device)  # Darknet 在 model.py 文件中，这里可以有个  img_size 参数可以配置输入数据的尺寸
    model.apply(weights_init_normal)  #  weights_init_normal  在 utils/utils.py 文件中，对权重进行初始化

    # If specified we start from checkpoint
    # 从预训练模型进行训练，判断是 torch 模型 pth 文件还是 Darknet 预训练权重，两种权重的加载方式不同
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))  
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # 数据加载器
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)  
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    
    
    # optimizer = torch.optim.Adam(model.parameters())
    
    # # 查看可优化的参数有哪些
    # for name, param in model.named_parameters():
    #   if param.requires_grad:
    #       print(name)
    
    """
    module_list.0.conv_0.weight
    module_list.0.batch_norm_0.weight
    module_list.0.batch_norm_0.bias
    module_list.1.conv_1.weight
    module_list.1.batch_norm_1.weight
    module_list.1.batch_norm_1.bias
    module_list.2.conv_2.weight
    module_list.2.batch_norm_2.weight
    module_list.2.batch_norm_2.bias
    module_list.3.conv_3.weight
    module_list.3.batch_norm_3.weight
    module_list.3.batch_norm_3.bias
    module_list.5.conv_5.weight
    module_list.5.batch_norm_5.weight
    module_list.5.batch_norm_5.bias
    module_list.6.conv_6.weight
    module_list.6.batch_norm_6.weight
    module_list.6.batch_norm_6.bias
    module_list.7.conv_7.weight
    module_list.7.batch_norm_7.weight
    module_list.7.batch_norm_7.bias
    module_list.9.conv_9.weight
    module_list.9.batch_norm_9.weight
    module_list.9.batch_norm_9.bias
    module_list.10.conv_10.weight
    module_list.10.batch_norm_10.weight
    module_list.10.batch_norm_10.bias
    module_list.12.conv_12.weight
    module_list.12.batch_norm_12.weight
    module_list.12.batch_norm_12.bias
    module_list.13.conv_13.weight
    module_list.13.batch_norm_13.weight
    module_list.13.batch_norm_13.bias
    module_list.14.conv_14.weight
    module_list.14.batch_norm_14.weight
    module_list.14.batch_norm_14.bias
    module_list.16.conv_16.weight
    module_list.16.batch_norm_16.weight
    module_list.16.batch_norm_16.bias
    module_list.17.conv_17.weight
    module_list.17.batch_norm_17.weight
    module_list.17.batch_norm_17.bias
    module_list.19.conv_19.weight
    module_list.19.batch_norm_19.weight
    module_list.19.batch_norm_19.bias
    module_list.20.conv_20.weight
    module_list.20.batch_norm_20.weight
    module_list.20.batch_norm_20.bias
    module_list.22.conv_22.weight
    module_list.22.batch_norm_22.weight
    module_list.22.batch_norm_22.bias
    module_list.23.conv_23.weight
    module_list.23.batch_norm_23.weight
    module_list.23.batch_norm_23.bias
    module_list.25.conv_25.weight
    module_list.25.batch_norm_25.weight
    module_list.25.batch_norm_25.bias
    module_list.26.conv_26.weight
    module_list.26.batch_norm_26.weight
    module_list.26.batch_norm_26.bias
    module_list.28.conv_28.weight
    module_list.28.batch_norm_28.weight
    module_list.28.batch_norm_28.bias
    module_list.29.conv_29.weight
    module_list.29.batch_norm_29.weight
    module_list.29.batch_norm_29.bias
    module_list.31.conv_31.weight
    module_list.31.batch_norm_31.weight
    module_list.31.batch_norm_31.bias
    module_list.32.conv_32.weight
    module_list.32.batch_norm_32.weight
    module_list.32.batch_norm_32.bias
    module_list.34.conv_34.weight
    module_list.34.batch_norm_34.weight
    module_list.34.batch_norm_34.bias
    module_list.35.conv_35.weight
    module_list.35.batch_norm_35.weight
    module_list.35.batch_norm_35.bias
    module_list.37.conv_37.weight
    module_list.37.batch_norm_37.weight
    module_list.37.batch_norm_37.bias
    module_list.38.conv_38.weight
    module_list.38.batch_norm_38.weight
    module_list.38.batch_norm_38.bias
    module_list.39.conv_39.weight
    module_list.39.batch_norm_39.weight
    module_list.39.batch_norm_39.bias
    module_list.41.conv_41.weight
    module_list.41.batch_norm_41.weight
    module_list.41.batch_norm_41.bias
    module_list.42.conv_42.weight
    module_list.42.batch_norm_42.weight
    module_list.42.batch_norm_42.bias
    module_list.44.conv_44.weight
    module_list.44.batch_norm_44.weight
    module_list.44.batch_norm_44.bias
    module_list.45.conv_45.weight
    module_list.45.batch_norm_45.weight
    module_list.45.batch_norm_45.bias
    module_list.47.conv_47.weight
    module_list.47.batch_norm_47.weight
    module_list.47.batch_norm_47.bias
    module_list.48.conv_48.weight
    module_list.48.batch_norm_48.weight
    module_list.48.batch_norm_48.bias
    module_list.50.conv_50.weight
    module_list.50.batch_norm_50.weight
    module_list.50.batch_norm_50.bias
    module_list.51.conv_51.weight
    module_list.51.batch_norm_51.weight
    module_list.51.batch_norm_51.bias
    module_list.53.conv_53.weight
    module_list.53.batch_norm_53.weight
    module_list.53.batch_norm_53.bias
    module_list.54.conv_54.weight
    module_list.54.batch_norm_54.weight
    module_list.54.batch_norm_54.bias
    module_list.56.conv_56.weight
    module_list.56.batch_norm_56.weight
    module_list.56.batch_norm_56.bias
    module_list.57.conv_57.weight
    module_list.57.batch_norm_57.weight
    module_list.57.batch_norm_57.bias
    module_list.59.conv_59.weight
    module_list.59.batch_norm_59.weight
    module_list.59.batch_norm_59.bias
    module_list.60.conv_60.weight
    module_list.60.batch_norm_60.weight
    module_list.60.batch_norm_60.bias
    module_list.62.conv_62.weight
    module_list.62.batch_norm_62.weight
    module_list.62.batch_norm_62.bias
    module_list.63.conv_63.weight
    module_list.63.batch_norm_63.weight
    module_list.63.batch_norm_63.bias
    module_list.64.conv_64.weight
    module_list.64.batch_norm_64.weight
    module_list.64.batch_norm_64.bias
    module_list.66.conv_66.weight
    module_list.66.batch_norm_66.weight
    module_list.66.batch_norm_66.bias
    module_list.67.conv_67.weight
    module_list.67.batch_norm_67.weight
    module_list.67.batch_norm_67.bias
    module_list.69.conv_69.weight
    module_list.69.batch_norm_69.weight
    module_list.69.batch_norm_69.bias
    module_list.70.conv_70.weight
    module_list.70.batch_norm_70.weight
    module_list.70.batch_norm_70.bias
    module_list.72.conv_72.weight
    module_list.72.batch_norm_72.weight
    module_list.72.batch_norm_72.bias
    module_list.73.conv_73.weight
    module_list.73.batch_norm_73.weight
    module_list.73.batch_norm_73.bias
    module_list.75.conv_75.weight
    module_list.75.batch_norm_75.weight
    module_list.75.batch_norm_75.bias
    module_list.76.conv_76.weight
    module_list.76.batch_norm_76.weight
    module_list.76.batch_norm_76.bias
    module_list.77.conv_77.weight
    module_list.77.batch_norm_77.weight
    module_list.77.batch_norm_77.bias
    module_list.78.conv_78.weight
    module_list.78.batch_norm_78.weight
    module_list.78.batch_norm_78.bias
    module_list.79.conv_79.weight
    module_list.79.batch_norm_79.weight
    module_list.79.batch_norm_79.bias
    module_list.80.conv_80.weight
    module_list.80.batch_norm_80.weight
    module_list.80.batch_norm_80.bias
    module_list.81.conv_81.weight
    module_list.81.conv_81.bias
    module_list.84.conv_84.weight
    module_list.84.batch_norm_84.weight
    module_list.84.batch_norm_84.bias
    module_list.87.conv_87.weight
    module_list.87.batch_norm_87.weight
    module_list.87.batch_norm_87.bias
    module_list.88.conv_88.weight
    module_list.88.batch_norm_88.weight
    module_list.88.batch_norm_88.bias
    module_list.89.conv_89.weight
    module_list.89.batch_norm_89.weight
    module_list.89.batch_norm_89.bias
    module_list.90.conv_90.weight
    module_list.90.batch_norm_90.weight
    module_list.90.batch_norm_90.bias
    module_list.91.conv_91.weight
    module_list.91.batch_norm_91.weight
    module_list.91.batch_norm_91.bias
    module_list.92.conv_92.weight
    module_list.92.batch_norm_92.weight
    module_list.92.batch_norm_92.bias
    module_list.93.conv_93.weight
    module_list.93.conv_93.bias
    module_list.96.conv_96.weight
    module_list.96.batch_norm_96.weight
    module_list.96.batch_norm_96.bias
    module_list.99.conv_99.weight
    module_list.99.batch_norm_99.weight
    module_list.99.batch_norm_99.bias
    module_list.100.conv_100.weight
    module_list.100.batch_norm_100.weight
    module_list.100.batch_norm_100.bias
    module_list.101.conv_101.weight
    module_list.101.batch_norm_101.weight
    module_list.101.batch_norm_101.bias
    module_list.102.conv_102.weight
    module_list.102.batch_norm_102.weight
    module_list.102.batch_norm_102.bias
    module_list.103.conv_103.weight
    module_list.103.batch_norm_103.weight
    module_list.103.batch_norm_103.bias
    module_list.104.conv_104.weight
    module_list.104.batch_norm_104.weight
    module_list.104.batch_norm_104.bias
    module_list.105.conv_105.weight
    module_list.105.conv_105.bias
    """

    # 冻结所有层: 
    for name, param in model.named_parameters(): 
        param.requires_grad = False

    # 解冻第105层
    for name, param in model.named_parameters(): 
      if '105' or '104' or '103' or '102'  or '101' in name:
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    
    
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
