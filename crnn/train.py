from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
#from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import torch.nn as nn
import copy
import models.crnn as crnn
import string

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet')
parser.add_argument('--expr_dir', default='output_results', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--deal_with_lossnan', action='store_true',help='whether to replace all nan/inf in gradients to zero')

opt = parser.parse_args()
print(opt)

#if not os.path.exists(opt.expr_dir):
    #os.makedirs(opt.expr_dir)
    #os.makedirs(opt.expr_dir+'/best')

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available() and opt.cuda:
    print('Nothing wrong with cuda')
    #opt.cuda = False

print(torch.cuda.is_available())
#print(opt.cuda)

def check_data(data_loader, name='sample'):
        data_iter = iter(data_loader)
        data = data_iter.next()
        cpu_images = data[0]
        cpu_texts = data[1]
        nim = min(32, cpu_images.size(0))
        out = torchvision.utils.make_grid(cpu_images[:nim], nrow=1)
        out = out.permute(1, 2, 0)
        out = (out*128 + 128).cpu().numpy()
        cv2.imwrite('/home/santhoshini/temp/{}.jpg'.format(name), out)

train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))

lexiconlist_filename = 'hindi.txt' 
p = open(lexiconlist_filename,'r').readlines()
opt.alphabet = p #string.digits + string.ascii_lowercase
# print(opt.alphabet)
nclass = len(opt.alphabet) + 1
nc = 1

print(nclass)

converter = utils.strLabelConverter(opt.alphabet)
criterion = nn.CTCLoss(zero_infinity=True)
#criterion = CTCLoss()

#print(converter.dict())
# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    crnn.load_state_dict(torch.load(opt.pretrained))
#print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.LongTensor(opt.batchSize)

#opt.cuda = False
if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

if opt.deal_with_lossnan:
    if torch.__version__ >= '1.1.0':
        criterion = nn.CTCLoss(zero_infinity = True)
    else:
        crnn.register_backward_hook(crnn.backward_hook)

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    crnn.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        #print(t)
        #print(l)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        #print("Predictions from the image crnn: \n")
        #print(preds.data)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        #print("Predictions size\n")
        #print(preds_size.data)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #print("Predictions after doing max:\n")
        #print(preds.data) 
        preds = preds.squeeze(1)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        #print("Predictoins after transposing")
        #print(preds.data)
        #print(preds_size.data)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            target = target.decode().lower()
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    print("Samples Correctly recognised =" + str(n_correct))
    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return accuracy


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    #print("Labels before encoding :")
    #print(cpu_texts)
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    #print("Labels are encoded into Tensors: ")
    #print(t)
    #print("Lengths- ")
    #print(l)
    #print('\n')
    utils.loadData(text, t)
    utils.loadData(length, l)
    #print(text)
 
    #print(length)
    #print('\n')
    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    #cost = criterion(nn.functional.log_softmax(preds), text, preds_size, length) / batch_size
    #print("Predictions : \n")
    #print(preds)
    #print("\n")
    cost = criterion(preds, text, preds_size, length) / batch_size
    #crnn.zero_grad()
    #print("Cost:")
    #print(cost)
    cost.backward()
    optimizer.step()
    return cost

losses_per_epoch = []
acc_per_epoch = []
best_acc = 0.0
is_best = 0
l_avg = utils.averager()
#list_bestlosses = []
#best_model = crnn.train()
#opt.nepoch = 25
for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        l_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            acc = val(crnn, test_dataset, criterion)
            acc_per_epoch.append(str(epoch) + " " + str(acc))
            is_best = acc >= best_acc
            if is_best:
                best_acc = acc

                filename = '{0}/best_model_{1}_'.format(opt.expr_dir, epoch) + '{0}'.format(best_acc*10000) + '.pth'
                torch.save(crnn.state_dict(), filename)
                is_best = 0
        # do checkpointing
        #if i % opt.saveInterval == 0:
        #    torch.save(
        #        crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))
    #l_avg.reset()
    #acc_per_epoch.append(str(best_acc))

g = open('/scratch/sanjana/valacc.txt','w')
s2 = '\n'.join(acc_per_epoch)
g.write(s2)
g.close()
