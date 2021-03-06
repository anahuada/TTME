import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayer
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim * num_heads) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([CustomGATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(CustomGATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        ########################################################################
        self.hidden_dim = hidden_dim * num_heads
        self.proj1 = nn.Linear(self.hidden_dim,self.hidden_dim**2) #baseline4
        self.proj2 = nn.Linear(self.hidden_dim,self.hidden_dim) #baseline4
        self.src_embedding_h = nn.Embedding(in_dim_node, self.hidden_dim) #baseline1
        self.dst_embedding_h = nn.Embedding(in_dim_node, self.hidden_dim) #baseline1
        self.edge_proj = nn.Linear(2*self.hidden_dim,self.hidden_dim) #baseline1
        self.edge_proj3 = nn.Linear(self.hidden_dim,self.hidden_dim) #baseline4
        
        self.bn_node_lr_e = nn.BatchNorm1d(self.hidden_dim) #baseline1~4

###########################################################################
### lr_g
        self.proj_g1 = nn.Embedding(in_dim_node,self.hidden_dim**2) #lr_g
        self.bn_node_lr_g1 = nn.BatchNorm1d(self.hidden_dim**2)
        self.proj_g2 = nn.Embedding(in_dim_node,self.hidden_dim) #lr_g
        self.bn_node_lr_g2 = nn.BatchNorm1d(self.hidden_dim)
        self.proj_g = nn.Linear(self.hidden_dim, 1)
        self.edge_thresh = 0.3 #0.5 #0.7
###########################################################################

    def forward(self, g, h, e):
###########################################################################
### learned graph list
###########################################################################
#        N = h.shape[0]
        lr_gs = []
        gs = dgl.unbatch(g)
        for g in gs:
            N = g.number_of_nodes()
            h_single = g.ndata['feat'].to(h.device)
            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.proj_g1(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.proj_g2(h_single))), 0.1, training=self.training).permute(1,0)
#            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.embedding_h(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
#            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.embedding_h(h_single))), 0.1, training=self.training).permute(1,0)

            mm = torch.mm(h_proj1,h_proj2)
            mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]            
        
            
            mm = self.proj_g(mm).squeeze(-1)
            diag_mm = torch.diag(mm)  # 取 a 对角线元素，输出为 1*3
            diag_mm = torch.diag_embed(diag_mm)
            mm -= diag_mm
            
#            matrix = torch.sigmoid(mm)
#            matrix = F.softmax(mm, dim=0)
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
            
    
            #binarized = BinarizedF()
            #matrix = binarized.apply(matrix) #(0/1)
            lr_connetion = torch.where(matrix>self.edge_thresh)

            #####################################
            #new_g = dgl.DGLGraph() # init a new graph
            #new_g.add_nodes(N)
            #new_g.ndata['feat'] = g.ndata['feat']
            #new_g.add_edges(lr_connetion[0], lr_connetion[1])
            #lr_gs.append(new_g)
            #####################################
            g.add_edges(lr_connetion[0], lr_connetion[1])
            lr_gs.append(g)
            
        g = dgl.batch(lr_gs).to(h.device)

#        import pdb; pdb.set_trace()
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
###########################################################################
###baseline 4
        g.apply_edges(lambda edges: {'src' : edges.src['feat']})
        src = self.src_embedding_h(g.edata['src'].to(h.device)) #[M,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['feat']})
        dst = self.dst_embedding_h(g.edata['dst'].to(h.device)) #[M,D]
        edge = torch.cat((src,dst),-1) #[M,2]
        lr_e_local = self.edge_proj(edge) #[M,D]

        N = h.shape[0]
        h_proj1 = self.proj1(h).view(-1,self.hidden_dim)
        h_proj2 = self.proj2(h).permute(1,0)
        mm = torch.mm(h_proj1,h_proj2)
        mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]
        lr_e_global = mm[g.all_edges()[0],g.all_edges()[1],:] #[M,D]
        lr_e_global = self.edge_proj3(lr_e_global) 
        
#        e = e + lr_e #baseline1-3
        e = lr_e_local + lr_e_global #baseline4        
        
        # bn=>relu=>dropout
        e = self.bn_node_lr_e(e)
        e = F.relu(e)
        e = F.dropout(e, 0.1, training=self.training)  

#        import pdb; pdb.set_trace()
        
        # GAT
        for conv in self.layers:
            h, e = conv(g, h, e)
            
        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss



        
