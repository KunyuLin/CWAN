from typing import Tuple, Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.grl import WarmStartGradientReverseLayer
from common.utils.metric import binary_accuracy
from ..modules.entropy import entropy

__all__ = ['DomainAdversarialLoss']


class CycleInconsistencyWeightModuleEuclidean(object):
    r"""
    Calculating class weight based on the output of discriminator.
    Introduced by `Importance Weighted Adversarial Nets for Partial Domain Adaptation (CVPR 2018) <https://arxiv.org/abs/1803.09210>`_
    """

    def __init__(self, softmax_temp: float, num_cluster: int, src_cluster: torch.Tensor, trg_cluster: torch.Tensor, 
                 momentum: float, partial_classes_index: Optional[List[int]] = None):
        self.softmax_temp = softmax_temp
        self.num_cluster = num_cluster
        self.src_cluster = src_cluster
        self.trg_cluster = trg_cluster
        self.momentum = momentum

    @torch.no_grad()
    def src_cluster_update(self, feature, cidx):
        sidx = torch.zeros(feature.shape[0], self.src_cluster.shape[0]).cuda()
        sidx = sidx.scatter(-1, cidx.unsqueeze(-1), 1).transpose(0,1)
        sidx /= sidx.sum(dim=1, keepdim=True)
        cset = cidx.unique()
        new_clu = torch.matmul(sidx[cset], feature)
        self.src_cluster[cset] = self.momentum * self.src_cluster[cset] + (1 - self.momentum) * new_clu

    @torch.no_grad()
    def trg_cluster_update(self, feature):
        num_fea = feature.shape[0]
        num_clu = self.trg_cluster.shape[0]
        fnm = feature.square().sum(dim=1).unsqueeze(0).expand(num_clu, num_fea) 
        cnm = self.trg_cluster.square().sum(dim=1).unsqueeze(1).expand(num_clu, num_fea) 
        dst = fnm + cnm - 2*torch.matmul(self.trg_cluster, feature.t())
        cidx = dst.min(dim=0)[1]

        sidx = torch.zeros(feature.shape[0], self.trg_cluster.shape[0]).cuda()
        sidx = sidx.scatter(-1, cidx.unsqueeze(-1), 1).transpose(0,1)
        sidx /= sidx.sum(dim=1, keepdim=True)
        cset = cidx.unique()
        new_clu = torch.matmul(sidx[cset], feature)
        self.trg_cluster[cset] = self.momentum * self.trg_cluster[cset] + (1 - self.momentum) * new_clu

    def xdomain_transform(self, feature, cluster):
        num_fea = feature.shape[0]
        num_clu = cluster.shape[0]
        fnm = feature.square().sum(dim=1).unsqueeze(1).expand(num_fea, num_clu) 
        cnm = cluster.square().sum(dim=1).unsqueeze(0).expand(num_fea, num_clu) 
        dst = fnm + cnm - 2*torch.matmul(feature, cluster.t())
        sim = F.softmax(-dst / self.softmax_temp, dim=1)
        fopp = torch.matmul(sim, cluster)
        return fopp

    def get_importance_weight(self, score, idx):
        weight = score[range(score.shape[0]), idx]
        weight = weight / weight.mean()
        weight = weight.detach()
        return weight


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


class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))



class ClassifierBase(nn.Module):
    """A generic Classifier class for domain adaptation.

    Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (int): Number of classes
        bottleneck (torch.nn.Module, optional): Any bottleneck layer. Use no bottleneck by default
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: -1
        head (torch.nn.Module, optional): Any classifier head. Use :class:`torch.nn.Linear` by default
        finetune (bool): Whether finetune the classifier or train from scratch. Default: True

    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride :meth:`~Classifier.get_parameters`.

    Inputs:
        - x (tensor): input data fed to `backbone`

    Outputs:
        - predictions: classifier's predictions
        - features: features after `bottleneck` layer and before `head` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, bias=True):
        super(ClassifierBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            print('Classifier bias = {}'.format(bias))
            self.head = nn.Linear(self._features_dim, num_classes, bias=bias)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        # f = self.bottleneck(x)
        predictions = self.head(x)
        return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        # f = self.bottleneck(x)
        predictions = self.head(x)
        return predictions

