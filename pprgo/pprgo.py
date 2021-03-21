from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter
from rgnn_at_scale.aggregation import ROBUST_MEANS

from .pytorch_utils import MixedDropout, MixedLinear


class PPRGoMLP(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False):
        super().__init__()
        self.use_batch_norm = batch_norm

        layers = [MixedLinear(num_features, hidden_size, bias=False)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))

        for i in range(nlayers - 2):
            layers.append(nn.ReLU())
            layers.append(MixedDropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

        layers.append(nn.ReLU())
        layers.append(MixedDropout(dropout))
        layers.append(nn.Linear(hidden_size, num_classes, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):

        embs = self.layers(X)
        return embs

    def reset_parameters(self):
        self.layers.reset_parameters()


class PPRGo(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout, batch_norm)
        self.aggr = aggr

    def forward(self,
                X: SparseTensor,
                ppr_scores: torch.Tensor,
                ppr_idx: torch.Tensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features for all nodes which were assigned a ppr score
            ppr_scores: torch.Tensor of shape (num_ppr_nodes)
                The ppr scores are calculate for every node of the batch individually.
                This tensor contains these concatenated ppr scores for every node in the batch.
            ppr_idx: torch.Tensor of shape (num_ppr_nodes)
                The id of the batch that the corresponding ppr_score entry belongs to

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        logits = self.mlp(X)
        propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce=self.aggr)
        return propagated_logits


class PPRGoEmmbeddingDiffusions(nn.Module):
    """
    Just like PPRGo, but diffusing/aggregating on the embedding space and not the logit space.
    """

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 skip_connection=False,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        # TODO: rewrite PPRGoMLP such that it doesn't expect at least n_layers >= 2.
        self.skip_connection = skip_connection

        layer_num_mlp = math.ceil(nlayers / 2)
        layer_num_mlp_logits = math.floor(nlayers / 2)

        if self.skip_connection:
            assert hidden_size > num_features, "hidden size must be greater than num_features for this skip_connection implementation to work"
            self.mlp = PPRGoMLP(num_features, hidden_size - num_features,
                                hidden_size, layer_num_mlp, dropout, batch_norm)
        else:
            self.mlp = PPRGoMLP(num_features, hidden_size,
                                hidden_size, layer_num_mlp, dropout, batch_norm)

        self.mlp_logits = PPRGoMLP(hidden_size, num_classes,
                                   hidden_size, layer_num_mlp_logits, dropout, batch_norm)
        self.aggr = aggr

    def forward(self,
                X: SparseTensor,
                ppr_scores: torch.Tensor,
                ppr_idx: torch.Tensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features for all nodes which were assigned a ppr score
            ppr_scores: torch.Tensor of shape (num_ppr_nodes)
                The ppr scores are calculate for every node of the batch individually.
                This tensor contains these concatenated ppr scores for every node in the batch.
            ppr_idx: torch.Tensor of shape (num_ppr_nodes)
                The id of the batch that the corresponding ppr_score entry belongs to

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        embedding = self.mlp(X)
        propagated_embedding = scatter(embedding * ppr_scores[:, None], ppr_idx[:, None],
                                       dim=0, dim_size=ppr_idx[-1] + 1, reduce=self.aggr)
        if self.skip_connection:
            # concatenated node features and propagated node embedding on feature dimension:
            propagated_embedding = torch.cat((X[ppr_idx.unique()], propagated_embedding), dim=-1)

        return self.mlp_logits(propagated_embedding)


class RobustPPRGo(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=32,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        super().__init__()
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout, batch_norm)

    def forward(self,
                X: SparseTensor,
                ppr_scores: SparseTensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features of all neighboring from nodes of the ppr_matrix (training nodes)
            ppr_matrix: torch_sparse.SparseTensor of shape (ppr_num_nonzeros, num_features)
                The node features of all neighboring nodes of the training nodes in
                the graph derived from the Personal Page Rank as specified by idx

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        logits = self.mlp(X)

        if self._mean.__name__ == 'soft_median' and ppr_scores.size(0) == 1 and 'temperature' in self._mean_kwargs:
            c = logits.shape[1]
            weights = ppr_scores.storage.value()
            with torch.no_grad():
                sort_idx = logits.argsort(0)
                weights_cumsum = weights[sort_idx].cumsum(0)
                median_idx = sort_idx[(weights_cumsum < weights_cumsum[-1][None, :] / 2).sum(0), torch.arange(c)]
            median = logits[median_idx, torch.arange(c)]
            distances = torch.norm(logits - median[None, :], dim=1) / pow(c, 1 / 2)

            soft_weights = weights * F.softmax(-distances / self._mean_kwargs['temperature'], dim=-1)
            soft_weights /= soft_weights.sum()
            new_logits = (soft_weights[:, None] * weights.sum() * logits).sum(0)

            return new_logits[None, :]

        if "k" in self._mean_kwargs.keys() and "with_weight_correction" in self._mean_kwargs.keys():
            # `n` less than `k` and `with_weight_correction` is not implemented
            # so we need to make sure we set with_weight_correction to false if n less than k
            if self._mean_kwargs["k"] > X.size(0):
                print("no with_weight_correction")
                return self._mean(ppr_scores,
                                  logits,
                                  # we can not manipluate self._mean_kwargs because this would affect
                                  # the next call to forward, so we do it this way
                                  with_weight_correction=False,
                                  ** {k: v for k, v in self._mean_kwargs.items() if k != "with_weight_correction"})
        return self._mean(ppr_scores,
                          logits,
                          **self._mean_kwargs)


class RobustPPRGoEmmbeddingDiffusions(nn.Module):
    """
    Just like RobustPPRGo, but diffusing/aggregating on the embedding space and not the logit space.
    """

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=32,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        super().__init__()
        # TODO: rewrite PPRGoMLP such that it doesn't expect at least n_layers >= 2.
        assert nlayers >= 4, "nlayers must be 4 or greater for this implementation to work"
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs
        self.mlp = PPRGoMLP(num_features, hidden_size,
                            hidden_size, nlayers - 2, dropout, batch_norm)

        self.mlp_logits = PPRGoMLP(hidden_size, num_classes,
                                   hidden_size, 2, dropout, batch_norm)

    def forward(self,
                X: SparseTensor,
                ppr_scores: SparseTensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features of all neighboring from nodes of the ppr_matrix (training nodes)
            ppr_matrix: torch_sparse.SparseTensor of shape (ppr_num_nonzeros, num_features)
                The node features of all neighboring nodes of the training nodes in
                the graph derived from the Personal Page Rank as specified by idx

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        embedding = self.mlp(X)

        if self._mean.__name__ == 'soft_median' and ppr_scores.size(0) == 1 and 'temperature' in self._mean_kwargs:
            c = embedding.shape[1]
            weights = ppr_scores.storage.value()
            with torch.no_grad():
                sort_idx = embedding.argsort(0)
                weights_cumsum = weights[sort_idx].cumsum(0)
                median_idx = sort_idx[(weights_cumsum < weights_cumsum[-1][None, :] / 2).sum(0), torch.arange(c)]
            median = embedding[median_idx, torch.arange(c)]
            distances = torch.norm(embedding - median[None, :], dim=1) / pow(c, 1 / 2)

            soft_weights = weights * F.softmax(-distances / self._mean_kwargs['temperature'], dim=-1)
            soft_weights /= soft_weights.sum()
            new_embedding = (soft_weights[:, None] * weights.sum() * embedding).sum(0)

            diffused_embedding = new_embedding[None, :]

        elif "k" in self._mean_kwargs.keys() and "with_weight_correction" in self._mean_kwargs.keys() \
                and self._mean_kwargs["k"] > X.size(0):
            # `n` less than `k` and `with_weight_correction` is not implemented
            # so we need to make sure we set with_weight_correction to false if n less than k
            print("no with_weight_correction")
            diffused_embedding = self._mean(ppr_scores,
                                            embedding,
                                            # we can not manipluate self._mean_kwargs because this would affect
                                            # the next call to forward, so we do it this way
                                            with_weight_correction=False,
                                            ** {k: v for k, v in self._mean_kwargs.items() if k != "with_weight_correction"})
        else:
            diffused_embedding = self._mean(ppr_scores,
                                            embedding,
                                            **self._mean_kwargs)
        return self.mlp_logits(diffused_embedding)
