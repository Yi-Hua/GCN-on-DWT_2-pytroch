from __future__ import division
from __future__ import print_function

import time
import argparse             # Argument Parser 參數解析器 : prog    usage    description    epilog  
import numpy as np

import torch
import torch.nn.functional as F  #F.nll_loss
import torch.optim as optim
from torch.autograd import Variable

#from pygcn.utils import load_data, accuracy, load_graph_data
#from pygcn.models import GCN
from utils import load_data, accuracy, load_graph_data
from models import GCN
torch.manual_seed(0)        # 設定初始random一致
# =================
# Training settings
# =================
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=3,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# =========
# Load data
# =========
lossfun = 1     #0:null_loss() ; 1:cross_entropy()  ; 2:MSE
n = 8

filename ='Catbox{}_xi_norm5%'.format(str(n))
imgs_data = np.array([load_data(image='../data/{}/H1.txt'.format(filename))])   # imgs_data = (feature, label)
adj = load_graph_data('../data/{}/Edge{}.txt'.format(filename,str(n)),'../data/{}/H1.txt'.format(filename))   # adj = (indices, values)

n_feature = imgs_data[0][0].shape[1]  # c = 3 (R,G,B)   # imgs_data[0][0]:feature ; imgs_data[0][1]: labels     # .shape[0] = rows num ;  .shape[1] = column num
n_class = 2     
col = 1

train_idx = range(0, 1)
test_idx = range(0, 1)

# ===================
# Model and optimizer
# ===================
model = GCN(nfeat=n_feature,
            nhid=args.hidden,
            nclass=n_class,
            dropout=args.dropout)

# model = GCN(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=labels.max() + 1,
#             dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
print('Model and optimizer finished !')
if args.cuda:
    model.cuda()
    feature = feature.cuda()
    adj = adj.cuda()
    label = label.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    print('')
    print('=============== Epoch: {:04d} ==============='.format(epoch+1))
    for idx, data in enumerate(imgs_data[train_idx]):
        #print('idx', idx)
        #print('data', type(data))
        feature, label = data 

        #idx = np.where(label.numpy() == 0)[0]
        idx = np.arange(n**2)
        idx = torch.from_numpy(idx).long()
        feature = feature[idx, :]
        label = label[idx]
        # print(feature)
        # print(label)
        # print('--- H0 ---', feature[:8,:8])
        feature = Variable(feature*10000)
        # print('--- H0 ---', feature[:8,:8])
        # print(feature[0:4,0:4])
        adj_max = Variable(adj)
        label = Variable(label)
        gc1, output = model(feature, adj_max)
        
        if epoch % 5 == 0:
            
            true_label = label.data.numpy()
            print(true_label)
            gcn_1 = gc1.data.numpy()
            # print(gcn_1.shape)
            with open('./gcn_1.npy', 'wb') as f:
                np.savez(f, feature=feature.data.numpy(), gcn_1=gcn_1, true_label=true_label)
            print('saved gc_1.npy')

    # print('output ', output.size())
    # prob = output.data.numpy().sum(0)     # D.append(prob)
    # print('----- output.sum -----', prob)
    # prob_map = output.data.numpy().reshape(1, 2, 8, 8)[:, 0, :, :]
    # print('output:',output[:3,:1])
    # print(output)
    # -------------
    # Loss Function
    # -------------
    if  lossfun == 0:
        loss_train = F.nll_loss(output, label)
    elif lossfun == 1:
        weights = [1., 1.] #[ 1 / number of instances for each class]
        class_weights = torch.FloatTensor(weights)
        loss_train = F.cross_entropy(output, label, weight=class_weights)
    elif lossfun == 2:
        loss_train = F.mse_loss(output, label) # torch.nn.MSELoss(output, label)

    acc_train = accuracy(output, label)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    # print('************epoch {:04d}************'.format(epoch+1))
    # print('-------- Before update weight --------')
    # print('W1 ', model.gc1.weight)
    # print('W2 ', model.gc2.weight)
    # print('W1 ', model.gc1.weight.grad)
    # print('W2 ', model.gc2.weight.grad)
    optimizer.step()
    # print('-------- After update weight --------')
    # print('W1 ', model.gc1.weight)
    # print('W2 ', model.gc2.weight)
    
    
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        gc1, output = model(feature, adj_max)

    # -------------
    # Loss Function
    # -------------
    if lossfun == 1:
        loss_val = F.cross_entropy(output, label, weight=class_weights)
    elif lossfun == 0:
        loss_val = F.nll_loss(output, label)
      
    acc_val = accuracy(output, label)
    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          #'acc_train: {:.4f}'.format(acc_train.data[0]),
          'acc_train: {}/{} {}/{}'.format(acc_train[0],acc_train[1],acc_train[2],acc_train[3]),
          #'loss_val: {:.4f}'.format(loss_val.data[0]),
          #'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    for feature, label in imgs_data[test_idx]:
        
        #output = model(feature, adj)
        feature = Variable(feature)
        adj_max = Variable(adj)
        label = Variable(label)
        output = model(feature, adj_max)
        # -------------
        # Loss Function
        # -------------
        if lossfun == 0:
            loss_test = F.nll_loss(output, label)
        elif lossfun == 1:
            loss_test = F.cross_entropy(output, label, weight=class_weights)

        acc_test = accuracy(output, label)
        # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        # acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data[0]),
              "accuracy= {:.4f}".format(acc_test.data[0]))
    

# ===========
# Train model
# ===========
t_total = time.time()
print("Optimization Stared!")
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# =======
# Testing
# =======
#test()
