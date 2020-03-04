from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import torchvision
from torch.utils import mkldnn
import time
import json
import pretrainedmodels

def main():
    parser = argparse.ArgumentParser(description='PyTorch Resnet50 benchmakr')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--data-size', type=int, default=100, metavar='N',
                        help='input data size (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--index', type=int, default=0),
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    model_name = 'se_resnext50_32x4d'
    #model = pretrainedmodels.__dict__[model_name](num_classes=1000, \
    #        pretrained='imagenet')
    #model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    #torch.save(model, "se_resnext50_32x4d_jd.pth")
    model = torch.load("se_resnext50_32x4d_jd.pth")
    #return
    model.eval()
    opt_mkldnn = True
    #opt_mkldnn = False 
    if opt_mkldnn:
        model = mkldnn.to_mkldnn(model)

    warmup_times = 10
    batch_size = args.batch_size
    data_size = args.data_size
    times = int(data_size/batch_size)
    w=h=320
    data = torch.rand(batch_size, 3, w, h)
    cnt = 0
    with torch.no_grad():
        for i in range(warmup_times):
            if opt_mkldnn:
                output = model(data.to_mkldnn())
            else:
                output = model(data)
		
        begin = time.time()
        for i in range(times):
            if opt_mkldnn:
                output = model(data.to_mkldnn())
            else:
                output = model(data)
        cnt = times
        if data_size % batch_size !=0:
            data = torch.rand(data_size % batch_size, 3, w, h)
            if opt_mkldnn:
                output = model(data.to_mkldnn())
            else:
                output = model(data)
            cnt += 1
        end = time.time()
    #data_size = len(test_loader.dataset)
    throughput =  data_size/(end - begin)
    latency = (end-begin)*1000/ cnt
    #print("Instance {}".format(args.index))
    #print("Image size {} {} {}".format(3, w, h))
    print("total time {} s".format(end-begin))
    #print("data size {}, batch size {}".format(data_size, batch_size))
    #print("throughput {} fps, latency {} ms".format(throughput, latency))
    res={}
    res['batch_size'] = batch_size
    res['data_size'] = data_size
    res['batch_shape'] = (batch_size, 3, w, h)
    res['latency'] = latency
    res['throughput'] = throughput

    res_file = "./log/res_{}.json".format(args.index)
    with open(res_file, "w") as f:
        f.write(json.dumps(res))
        #print("save {} with {}".format(res_file, res))
if __name__ == '__main__':
    main()
