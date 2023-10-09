from __future__ import absolute_import
import sys
sys.path.append('./')

import argparse
import os
import os.path as osp
import numpy as np
import math
import time
from PIL import Image, ImageFile

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
#import six
import sys
from PIL import Image
import numpy as np
import io

import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_args
from lib import datasets, evaluation_metrics, models
from lib.models.model_builder import ModelBuilder
from lib.datasets.dataset import LmdbDataset, AlignCollate
from lib.loss import SequenceCrossEntropyLoss
from lib.trainers import Trainer
from lib.evaluators import Evaluator
from lib.utils.logging import Logger, TFLogger
from lib.utils.serialization import load_checkpoint, save_checkpoint
from lib.utils.osutils import make_symlink_if_not_exists
from lib.evaluation_metrics.metrics import get_str_list
from lib.utils.labelmaps import get_vocabulary, labels2strs

global_args = get_args(sys.argv[1:])

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=2,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0) 

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= self.nSamples, 'index range error'
        #print(self.nSamples)
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(str(img_key).encode())
            #print(type(imgbuf))  
            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            
            try:
                img = Image.open(buf)
                img = img.convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)
    
def image_process(img, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    #img = Image.open(image_path).convert('RGB')
    
    if keep_ratio:
        w, h = img.size
        ratio = w / float(h)
        imgW = int(np.floor(ratio * imgH))
        imgW = max(imgH * min_ratio, imgW)

    img = img.resize((imgW, imgH), Image.BILINEAR)
    img = transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)

    return img

class DataInfo(object):
  """
  Save the info about the dataset.
  This a code snippet from dataset.py
  """
  def __init__(self, voc_type):
    super(DataInfo, self).__init__()
    self.voc_type = voc_type

    assert voc_type in ['INDIA','LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))
    self.rec_num_classes = len(self.voc)

def main(args):
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('using cuda.')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
  
    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (32, 100)

    dataset_info = DataInfo(args.voc_type)

    # Create model
    model = ModelBuilder(arch=args.arch, rec_num_classes=dataset_info.rec_num_classes,
                       sDim=args.decoder_sdim, attDim=args.attDim, max_len_labels=args.max_len,
                       eos=dataset_info.char2id[dataset_info.EOS], STN_ON=args.STN_ON)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)
        
    nwrr = 0
    norm_ED = 0

    # Evaluation
    model.eval()
    lmdb_data = '/content/gdrive/MyDrive/CRNN_Data/gujarati_val_lmdb/'
    lm = lmdbDataset(lmdb_data)
    p = lm.__len__()
    print(p)
    for i in range(0,p):
        img, gt = lm.__getitem__(i)
        img = image_process(img)
        with torch.no_grad():
            img = img.to(device)
            
        if i%1000 == 0:
            print("Finished testing - " + str(i) + " images...")
        
        input_dict = {}
        input_dict['images'] = img.unsqueeze(0)
        # TODO: testing should be more clean.
        # to be compatible with the lmdb-based testing, need to construct some meaningless variables.
        rec_targets = torch.IntTensor(1, args.max_len).fill_(1)
        rec_targets[:,args.max_len-1] = dataset_info.char2id[dataset_info.EOS]
        input_dict['rec_targets'] = rec_targets
        input_dict['rec_lengths'] = [args.max_len]
        output_dict = model(input_dict)
        pred_rec = output_dict['output']['pred_rec']
        pred_str, _ = get_str_list(pred_rec, input_dict['rec_targets'], dataset=dataset_info)
        #print('Recognition result: {0}'.format(pred_str[0]))
        pred = pred_str.strip()
        
        if(pred.strip() == gt.strip()):
            nwrr += 1
        if len(gt) == 0 or len(pred) == 0:
            norm_ED += 0
        elif len(gt) > len(pred):
            norm_ED += 1 - edit_distance(pred, gt) / len(gt)
        else:
            norm_ED += 1 - edit_distance(pred, gt) / len(pred)

    wrr = nwrr / float(p) * 100
    crr = norm_ED / float(p) 
    
    return wrr, crr
    
if __name__ == '__main__':
    # parse the config
    args = get_args(sys.argv[1:])
    main(args)

