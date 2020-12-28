import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .pytorch_utils import MixedDropout, MixedLinear
from rgnn_at_scale.aggregation import ROBUST_MEANS, chunked_message_and_aggregate


class PPRGoMLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout):
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
    def __init__(self, num_features, num_classes, hidden_size, nlayers, dropout, aggr="sum"):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout)
        self.aggr = aggr

    def forward(self, X, ppr_scores, ppr_idx):
        """
        Parameters: 
            X: torch.SparseTensor of shape (num_ppr_nodes, num_features)
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
