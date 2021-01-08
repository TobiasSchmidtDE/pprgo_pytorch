from rgnn_at_scale.aggregation import ROBUST_MEANS, chunked_message_and_aggregate
from .pytorch_utils import MixedDropout, MixedLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


class PPRGoMLP(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float):
        super().__init__()

        fcs = [MixedLinear(num_features, hidden_size, bias=False)]
        for i in range(nlayers - 2):
            fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
        fcs.append(nn.Linear(hidden_size, num_classes, bias=False))
        self.fcs = nn.ModuleList(fcs)

        self.drop = MixedDropout(dropout)

    def forward(self, X):
        embs = self.drop(X)
        embs = self.fcs[0](embs)
        for fc in self.fcs[1:]:
            embs = fc(self.drop(F.relu(embs)))
        return embs


class PPRGo(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout)
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


class RobustPPRGo(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=32,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        super().__init__()
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout)

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
