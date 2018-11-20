# Yolo-v1-pytorch-implementation
### Dependency:
```sh
pytorch 0.4.0, torchvision, opencv3, numpy, pandas, argparse
```
I recommend to use anaconda for python virtual environment.    
The code analysis and technical detail please see my note in my blog [YoloV1](https://duanyiqun.github.io/2018/11/12/yolo-literal-review-and-implementation/)   
To simplely run this demo please enter the pytorch environment and run this command:
```sh
python train.py -parameters
```
or 
```sh
nohup python train.py -parameters
```
The parameters have been preseted inside the py file. The training log will be saved in ./train/mname/xxx. And the Usage is as follows:

```sh
usage: train.py [-h] [--lr LR] [--resume] [--experimentname EXPERIMENTNAME]
                [--trainroot TRAINROOT] [--testroot TESTROOT]
                [--indexdir INDEXDIR] [--batch_size BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--sgrid SGRID]
                [--bbxnumber BBXNUMBER] [--classnumber CLASSNUMBER]
                [--mname MNAME]

PyTorch Yolov1 Training DuanYiqun

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --resume, -r          resume from checkpoint
  --experimentname EXPERIMENTNAME
                        model name for save
  --trainroot TRAINROOT
                        for train images
  --testroot TESTROOT   for test images
  --indexdir INDEXDIR   log direction for save
  --batch_size BATCH_SIZE
                        batch size
  --num_epochs NUM_EPOCHS
                        training length
  --sgrid SGRID         grid number 7*7 for default
  --bbxnumber BBXNUMBER
                        bounding box number
  --classnumber CLASSNUMBER
                        class number default is 20
  --mname MNAME         experimentname
```
Same to most of other implementations, you should firstly run [xml2txt.py](https://github.com/DuanYiqun/pytorch_implementation_of_Yolov1/blob/master/xml_2_txt.py) to change xml annotaitons of VOC dataset to a txt file which contains bounding boxes and classes line by line.    
This repository is writted as review, part of functions still under construction due to limited time, but the part for training has already completed and tested. 
Part of the functions in dataset.py and yololoss.py referenced to several existed blogs. 
