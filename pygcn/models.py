import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
       

    def forward(self, x, adj):
        # print('--- H0 ---', x[:8,:8])
        gc1 = self.gc1(x, adj)
        gc1 = F.relu(gc1)  
        # gc2 = self.gc2(gc1, adj)
        # fc1 = self.fc1(x)
        # fc1 = F.relu(fc1)
        fc2 = self.fc2(gc1)
        # print('gc1 x:', gc1[:8,:8])
        
        # fc1 = F.relu(fc1)
        # print('--- H1 ---', gc1[:8,:8])
      
        # fc2 = self.fc2(fc1)
        # print('--- H2 ---', gc2[:8,:8])
        # pred = self.fc(x)
        # pred = self.gc(x, adj)

        out = F.softmax(fc2)      #F.log_softmax(_)
        # print('--- softmax ---', out[:8,:8])
        return [gc1, out]

