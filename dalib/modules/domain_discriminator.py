from typing import List, Dict
import torch.nn as nn

__all__ = [
    'DomainDiscriminator',
    'MultiLayerDomainDiscriminator',
]


class DomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]


class MultiLayerDomainDiscriminator(nn.Sequential):
    r"""Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, in_feature: int, hidden_size: int, num_layers: int=2, batch_norm=True):
        if batch_norm:
            module_list = [
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            ]
            for _ in range(num_layers-1):
                module_list.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                ])
            module_list.extend([nn.Linear(hidden_size, 1), nn.Sigmoid()])
            super(MultiLayerDomainDiscriminator, self).__init__(*module_list)
        else:
            module_list = [
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            ]
            for _ in range(num_layers-1):
                module_list.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                ])
            module_list.extend([nn.Linear(hidden_size, 1), nn.Sigmoid()])
            super(MultiLayerDomainDiscriminator, self).__init__(*module_list)

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]


