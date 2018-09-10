import numpy as np          # NumPy(Numeric Python) : 加入陣列和矩陣與一些相當簡單的功能，ex陣列搜尋和排序陣列。
import scipy.sparse as sp   # SciPy(scientific Python) : 加入中級和進階函式，使用儲存在陣列和矩陣中的資料。
import torch
import time
from collections import OrderedDict


def encode_onehot(label):
    classes = list(set(label))
    classes.sort()
    classes_dict = OrderedDict() #排序
    for i, c in enumerate(classes):# classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        classes_dict[c] = np.identity(len(classes))[i, :]

    label_onehot = np.array(list(map(classes_dict.get, label)),
                             dtype = np.int32)
    return label_onehot
    
def load_graph_data(adj_file, img_file):    # Edge64.txt, H1.txt
    print('Loading {} adj_file...'.format(adj_file))
    t = time.time()

    # =====================================
    # Load : network graph structure (tree)
    # =====================================
    idx_H = np.genfromtxt(img_file, dtype = np.dtype(str))        # np.genfromtxt(): 創建數組表格數據 ; dtype(Data type)
    # idx_H = [idx, feature, label]
    idx = np.array(idx_H[:, 0], dtype=np.int32) # Node
    feature = sp.csr_matrix(idx_H[:, 1:-1], dtype = np.float32)
    label = encode_onehot(idx_H[:, -1])

    # =================
    # build graph (adj)
    # =================
    idx_map = {j: i for i, j in enumerate(idx)}  # 給每個點編序號:  11:0, 12:1, 13:2, ...
    edges_unordered = np.genfromtxt(adj_file, dtype=np.int32)   # 讀取adj_file
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                      dtype=np.int32).reshape(edges_unordered.shape) # adj_file 中的點換成序號  
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape = (label.shape[0], label.shape[0]),
                         dtype = np.float32)        # adj中為1.0的位置: (0,4)  1.0 , (0,5)  1.0 ,...
    #---coo_matrix((data, (row, col)), shape, dtype):行, 列存了data，其余位置皆为0---
    print('Build graph finished ... time: {:.4f}s'.format(time.time() - t))

    # ================================
    # build symmetric adjacency matrix
    # ================================
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) #按順序排
    # print(adj)
    adj = adj + sp.eye(adj.shape[0])
    # print('A+I: ',adj)
    adj = normalize(adj) # normalize matrix A' = (A+I)
    # print('normalize:',adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj) #adj = (indices, values)
    # print('adj_tensor:',adj)

    print('To_torch_sparse_tensor finished ... time: {:.4f}s'.format(time.time() - t))

    return idx_H, feature, label



def normalize(mx):
    """Row-normalize sparse matrix"""
    print('mx = ', mx)
    rowsum = np.array(mx.sum(1))
    print('mx.sum(1) = ', mx.sum(1))
    print('rowsum = np.array(mx.sum(1)) = ', rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



idx_H, feature, label = load_graph_data('../data/Test/Edge.txt','../data/Test/H.txt')
# print('-- feature and labels -- ', type(idx_H), 'H = ')
# print(idx_H[0:3])
# print('-- feature --', type(feature))
# print(feature)
# print('-- label -- ', type(label))  # type(label): <class 'tuple'>
# print(label)    #(array([[1, 0], [0, 1], [0, 1],...)

#label_onehot, classes , classes_dict = encode_onehot(idx_H[:, -1])
# print('---idx_H[:, -1]---')
# print(idx_H[:, -1])
# print('---label_onehot---', type(label_onehot)) # type(label_onehot): <class 'numpy.ndarray'>
# print(label_onehot)

