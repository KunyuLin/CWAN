from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.grl import WarmStartGradientReverseLayer
from common.utils.metric import binary_accuracy
from ..modules.entropy import entropy

__all__ = ['DomainAdversarialLoss']


class ClusterModuleEuclidean(object):

    def __init__(self, num_cluster: int, src_cluster: torch.Tensor, momentum: float, 
            partial_classes_index: Optional[List[int]] = None):
        self.num_cluster = num_cluster
        self.src_cluster = src_cluster
        self.momentum = momentum

    @torch.no_grad()
    def src_cluster_update(self, feature, cidx):
        sidx = torch.zeros(feature.shape[0], self.src_cluster.shape[0]).cuda()
        sidx = sidx.scatter(-1, cidx.unsqueeze(-1), 1).transpose(0,1)
        sidx /= sidx.sum(dim=1, keepdim=True)
        cset = cidx.unique()
        new_clu = torch.matmul(sidx[cset], feature)
        self.src_cluster[cset] = self.momentum * self.src_cluster[cset] + (1 - self.momentum) * new_clu



class ScoreEntropyWeightModule(object):

    def get_importance_weight(self, score):
        weight = 1.0 + torch.exp(-entropy(score))
        weight = weight / weight.mean()
        weight = weight.detach()
        return weight


def center_loss(feature, cluster, idx=None):
    num_fea = feature.shape[0]
    num_clu = cluster.shape[0]
    # use ground-truth or nearest cluster idx
    if idx is not None:
        pass
    else:
        fnm = feature.square().sum(dim=1).unsqueeze(1).expand(num_fea, num_clu) 
        cnm = cluster.square().sum(dim=1).unsqueeze(0).expand(num_fea, num_clu) 
        dst = fnm + cnm - 2*torch.matmul(feature, cluster.t())
        idx = dst.argmin(dim=1)
    # mse
    clu = torch.index_select(cluster, 0, idx)
    mse = F.mse_loss(feature, clu)
    return mse


