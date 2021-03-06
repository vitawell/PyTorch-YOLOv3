## 水下目标识别实践：
使用URPC比赛的水下图像数据训练已经在coco上预训练过的yolov3模型，并实现视频中的水下目标识别。

本项目主要参考 https://github.com/FLyingLSJ/PyTorch-YOLOv3-master

视频检测的实现参考 https://github.com/ayooshkathuria/pytorch-yolo-v3

```
主要为video_demo.py :注意运行时需要使用命令行或者修改参数中的视频名称和自己训练的模型checkpoints。

在colab上训练得到的模型为cuda版本，只能在安装了cuda的colab上使用。在自己笔记本上只能使用非cuda版本的模型？之后好像能用了。
```


### 图片未预处理：

##### 第一次
用了80张图片为训练集，20张图片为验证集，跑1个epoch大概20min，跑了几十个epoch，损失大概降为10。效果不好，同一个物体会识别为几个类，且无关物体也会识别。

##### 第二次
用了800张图片为训练集，200张图片为验证集，跑1个epoch大概1个小时，跑了3个epoch，损失大概降为5。效果好了点，但还是有同一个物体会识别为几个类，且无关物体也会识别。

##### 使用了Google Colab Notebook来训练模型
Colab文件：https://colab.research.google.com/drive/1Xq2cpXjVgxx_dMNSgmF4MDvQ9NLyVt7e?usp=sharing

跑一个epoch大概十几分钟，跑了10个epoch，损失大概降为3，效果提升不大。

又在这基础上只解冻101-105，跑了10个epoch，损失降为1.5，但是效果提升不大。

&emsp;

### 做点其他修改：
#### 1.冻结前面的层
（修改了train.py，但还没懂是只解冻了最后一层还是最后一个节点。跑了二十个epoch后，损失还是20左右。检测效果一般。可能是只训练一层效果提升不大。）

重新从coco预训练模型开始训练，修改为只解冻最后一层（105层），训练了10个epoch后损失还是35左右。
修改为只解冻101-105层，训练了10个epoch后损失变为3。图片效果没有无冻结跑20个epoch效果好。

可能针对水下图像，预训练模型的基本特征对检测效果帮助不大。

`修改为解冻全部层，然后冻结101-105层，出现错误：optimizer got an empty parameter list。尝试解决失败，放弃此种尝试。`

#### 2.去除置信度不高的类别
修改detect.py，加了一个if判断，只有类别置信度高于0.7(可输入指定别的阈值)的类别才输出框。
检测图片的结果效果提升明显，一个物体只会输出一个类别的框。

修改video_demo.py，加if判断，只有置信度高于阈值（默认为0.7，设置为0.8时效果比较好）的类别才输出框。
修改后效果提升明显，无关物体少了很多，但是准确度不够，而且也会出现误判类别。

#### 3.batch_size设置为16？（默认为8）

`colab中训练时若提示GPU不足，可能为batch_size过大。也可能是优先级变小，需终止会话后重新打开。`

在解冻101-105层的基础上训练了10个epoch，损失还是3，看来batch_size影响不大。


&emsp;

### 图像预处理：Retinex图像增强算法
用MSR处理了1000张水下图像，用其中800张图片作为训练集，200张图片为验证集。

MSR图像增强算法的Google Colab Notebook文件：https://colab.research.google.com/drive/1KIJJ6eUqfWzizU9eZmzxH7A3KZ2ZLPwg?usp=sharing

##### 第一次
只解冻101-105层，训练了个40个epoch，损失降为3，mAP最高为23%.

##### 第二次
在之前解冻所有层训练的基础上，只解冻101-105层，训练了10个epoch，损失降为3，mAP最高为22%。

对用MSR处理过的图片检测效果一般，对未经处理的图片检测效果更差一点。

##### 第三次
在第二次训练基础上解冻所有层，训练了35个epoch，损失降为1，mAP最高为26%.

对未经处理过的图片检测效果一般，对经MSR处理的图片检测效果也差不多。


&emsp;

&emsp;
————————————
## 以下为主要参考项目的readme：

本项目参考 https://github.com/eriklindernoren/PyTorch-YOLOv3 ，感谢大佬开源，在此基础上增加了数据准备的说明，项目流程说明。

[数据准备说明文档]: data/custom/readme.md	"数据准备说明文档"


`注意：因为本人的电脑没有 GPU ，故跑不了代码，本项目所有代码都是在 Linux / mac 下运行的，如果你的电脑（Win 操作系统想要运行本代码，可以安装 Git 软件来执行 Linux 命令）`



步骤如下：

### 1. 修改配置文件和下载预训练权重

```bash
$ cd config/   # Navigate to config dir
# Will create custom model 'yolov3-custom.cfg'
$ bash create_custom_model.sh <num-classes>   #  <num-classes> 类别数目参数，根据你的需要修改

# 下载预训练权重
$ cd weights/
$ bash download_weights.sh
```

### 2. 修改 config/custom.data 文件

```python
classes= 2  # 类别数，根据你的需要修改
train=data/custom/train.txt
valid=data/custom/valid.txt
names=data/custom/classes.names
```


### 3. 修改 data/custom/classes.names 文件

这里需要分两种情况

- 有完整的数据，也就是有原始的图片和 txt 文件的，请执行下面的操作

  因为你已经有了 txt  格式的标注数据了，大致的内容如下，第一个数字就是类别对应的代码了，比如 cat 对应 0，dog 对应 1，pig 对应 2

  ```bash
  0 0.014656616415410386 0.41642011834319526 0.024288107202680067 0.051775147928994084
  1 0.22989949748743718 0.33357988165680474 0.01423785594639866 0.034023668639053255
  2 0.28936348408710216 0.30029585798816566 0.01256281407035176 0.03254437869822485
  ```

  那么 data/custom/classes.names 文件必须写成以下形式，必须要对应，否则会出错的

  ```bash
  cat
  dog
  pig
  
  ```

  

- 没有完整的数据的，即，只有原始图片和 xml 文件的，请执行下面的操作

  每个类别一行，顺序要和 data/custom/3_trans.py 中的 classes 变量的顺序一样

不管你有么有完整的数据，也请看看第 4 步，没有完整数据的，根据第 4 步进行数据格式转换，有完整数据的，请看看数据的存放格式是怎么样的

### 4. 数据集处理流程请见 data/custom/readme.md



### 5. 上述准备准备完毕后开始训练

```bash
pip3 install -r requirements.txt
pip install terminaltables
```


```python
# 训练命令
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74
# 添加其他参数请见 train.py 文件。 !最后的模型名称改为yolov3-custom.cfg
    
# 从中断的地方开始训练
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights checkpoints/yolov3_ckpt_299.pth --epoch 
# checkpoints是保存的训练过的模型权重！ 末尾的--epoch 后面需加上数字，表示训练几轮。

```

```python
# 测试：
python detect.py --image_folder data/samples/ --weights_path checkpoints/yolov3_ckpt_25.pth --model_def config/yolov3-custom.cfg --class_path data/custom/classes.names
# 若是在 GPU 的电脑上训练，在 CPU 的电脑上预测，则需要修改 model.load_state_dict(torch.load(opt.weights_path, map_location='cpu'))
```


​    

- 本项目参考：https://github.com/eriklindernoren/PyTorch-YOLOv3
- yolo 博客地址：https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
- 机器之心翻译：https://www.jiqizhixin.com/articles/2018-04-23-3
- yolo源码解析：https://zhuanlan.zhihu.com/p/49981816
- yolo 解读：https://zhuanlan.zhihu.com/p/76802514

### 6. 其他

出现警告解决方案
`UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead`. 
在 model.py  **计算损失的位置 大概在 196 行左右**添加以下两句（我已经添加到源码中了）

```python 
obj_mask=obj_mask.bool() # convert int8 to bool
noobj_mask=noobj_mask.bool() #convert int8 to bool
```



注意预测的时候需要检查数据的格式问题（单通道？三通道？）



预测的值分别是  x1, y1, x2, y2, conf, cls_conf, cls_pred

cfg 中的 路由层（Route）
它的参数 layers 有一个或两个值。当只有一个值时，它输出这一层通过该值索引的特征图。
在我们的实验中设置为了 - 4，所以层级将输出路由层之前第四个层的特征图。
当层级有两个值时，它将返回由这两个值索引的拼接特征图。在我们的实验中为 - 1 和 61，
因此该层级将输出从前一层级（-1）到第 61 层的特征图，并将它们按深度拼接。
