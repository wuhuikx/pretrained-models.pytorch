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
import os

def main():
    parser = argparse.ArgumentParser(description='PyTorch Resnet50 benchmakr')
    parser.add_argument('--index-size', type=int, default=1),
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    latency = 0.0
    throughput = 0.0
    cnt = 0
    for i in range(args.index_size):
        res_file = "res_{}.json".format(i)
        if not os.path.exists(res_file):
            print("miss json for {}".format(i))
            cnt += 1
            continue
        with open(res_file, "r") as f:
            res = json.load(f)
            latency += res['latency']
            throughput += res['throughput']

    latency = latency/(args.index_size-cnt)
    throughput = throughput/(args.index_size-cnt)*args.index_size
    print("Miss {} instance result".format(cnt))

    print("Instance {}, latency {:1,.2f} ms, throughput {:2,.2f} fps".format(args.index_size, latency, throughput))
if __name__ == '__main__':
    main()
