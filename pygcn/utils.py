import numpy as np          # NumPy(Numeric Python) : 加入陣列和矩陣與一些相當簡單的功能，ex陣列搜尋和排序陣列。
import scipy.sparse as sp   # SciPy(scientific Python) : 加入中級和進階函式，使用儲存在陣列和矩陣中的資料。
import torch
import time
from collections import OrderedDict

def encode_onehot(label):
    classes = ['No','Yes']
    # classes = list(set(label))
    # classes.sort() # 'No','Yes'
    
    classes_dict = OrderedDict() #排序
    for i, c in enumerate(classes):# classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        classes_dict[c] = np.identity(len(classes))[i, :]

    label_onehot = np.array(list(map(classes_dict.get, label)),
                             dtype = np.int32)
    return label_onehot

def load_graph_data(adj_file, img_file):    # Edge64.txt, H1.txt
    # =====================================
    # Load : network graph structure (tree)
    # =====================================
    print('Loading {} adj_file...'.format(adj_file))
    
    t = time.time()
    print("")

    idx_H = np.genfromtxt(img_file, dtype = np.dtype(str))        # np.genfromtxt(generator from txt): 創建數組表格數據 ; dtype(Data type)
    # idx_H = [idx, feature, label]
    idx = np.array(idx_H[:, 0], dtype=np.int32)                   # Node
    # feature = sp.csr_matrix(idx_H[:, 1:-1], dtype = np.float32)   # class csr_matrix: 壓缩(cs)行格式，保留列(r) <-> csc_matrix: 壓缩(cs)列格式，保留行(c)
    label = encode_onehot(idx_H[:, -1])
    
    # ===========
    # build graph
    # ===========
    idx_map = {j: i for i, j in enumerate(idx)}  # 給每個點編序號:  11:0, 12:1, 13:2, ...
    edges_unordered = np.genfromtxt(adj_file, dtype=np.int32)   # 讀取adj_file
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                      dtype=np.int32).reshape(edges_unordered.shape) # adj_file 中的點換成序號
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape = (label.shape[0], label.shape[0]),
                         dtype = np.float32)        # adj中為1.0的位置: (0,4)  1.0 , (0,5)  1.0 ,...
    # print(adj)
    #---coo_matrix((data, (row, col)), shape, dtype):行, 列存了data，其余位置皆为0---
    print('Build graph finished ... time: {:.4f}s'.format(time.time() - t))

    # ================================
    # build symmetric adjacency matrix
    # ================================
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #按順序排
    adj = normalize(adj + sp.eye(adj.shape[0])) # normalize matrix A' = (A+I) 
    adj = sparse_mx_to_torch_sparse_tensor(adj)# adj = (indices, values) [torch.LongTensor of size 2x*],[torch.FloatTensor of size *]
    # print('----normalize adj----',adj)
    # print('=========sparse_mx_to_torch_sparse_tensor=========',adj)
    print('To_torch_sparse_tensor finished ... time: {:.4f}s'.format(time.time() - t))
    return adj


#def load_data(path="../data/cora/", dataset="cora"):
def load_data(image):
    print('Loading {} dataset...'.format(image))
    
    t = time.time()
    
    try:
        idx_H = np.genfromtxt(image, dtype=np.dtype(str))
        feature = sp.csr_matrix(idx_H[:, 1:-1], dtype = np.float32)  # class csr_matrix: 壓缩(cs)行格式，保留列(r) <-> csc_matrix: 壓缩(cs)列格式，保留行(c)
    
    except:
        print('image ', image)
        n_row, n_col = idx_H.shape
        for i in range(n_row):
            for j in range(1,3+1):
                try:
                    float(idx_H[i, j])
                except:
                    idx_H[i, j] = '0.0e+00'
        feature = sp.csr_matrix(idx_H[:, 1:-1], dtype = np.float32)
    # print('-------------------1-----------------', torch.FloatTensor(np.array(feature[0:4,0:4])))
    label = encode_onehot(idx_H[:, -1])
    label = torch.LongTensor(np.where(label)[1])

    # build symmetric adjacency matrix
    # feature = normalize(feature)
    # print('-------------------2-----------------',torch.FloatTensor(np.array(feature[0:4,0:4])))
    feature = torch.FloatTensor(np.array(feature.todense()))
    # print('-------------------3-----------------',feature[0:4,0:4])
    print('To_torch_sparse_tensor finished ... time: {:.4f}s'.format(time.time() - t))
    return feature, label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, label):   
    #preds = output.max(1)[1].type_as(label)
    #correct = preds.eq(label).double()
    output = output.max(1)[1].type_as(label).data.numpy()
    label = label.data.numpy()
    idx_0 = (label == 0)
    idx_1 = (label == 1)
    correct_0 = np.equal(output[idx_0], label[idx_0])
    correct_0 = correct_0.sum()
    
    correct_1 = np.equal(output[idx_1], label[idx_1])
    correct_1 = correct_1.sum()

    #return correct / sum(idx)
    return (correct_0, sum(idx_0), correct_1, sum(idx_1))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
