import torch

from .pytorch_utils import matrix_to_torch


class PPRDataset(torch.utils.data.Dataset):
    def __init__(self, attr_matrix_all, ppr_matrix, indices, labels_all=None):
        """
        Parameters:
            attr_matrix_all: np.ndarray of shape (num_nodes, num_features)
                Node features / attributes of all nodes in the graph
            ppr_matrix: scipy.sparse.csr.csr_matrix of shape (num_train_nodes, num_nodes)
                The personal page rank vectors for all nodes of the training set
            indices: np.ndarray of shape (num_train_nodes)
                The ids of the training nodes
            labels_all: np.ndarray of shape (num_nodes)
                The class labels for all nodes in the graph        
        """
        self.attr_matrix_all = attr_matrix_all
        self.ppr_matrix = ppr_matrix
        self.indices = indices
        self.labels_all = torch.tensor(labels_all)
        self.cached = {}

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        """
        Parameters:
            idx: List[Int] of shape (batch_size)
                The ids of the indicies array that point to the training nodes
        Returns:
            A touple (data, labels), where
                data: touple of
                    - attr_matrix_all: torch.SparseTensor of shape (ppr_num_nonzeros, num_features)
                        The node features of all neighboring nodes of the training nodes in 
                        the graph derived from the Personal Page Rank as specified by idx
                    - ppr_scores: torch.Tensor of shape (ppr_num_nonzeros)
                        The page rank scores of all neighboring nodes of the training nodes in 
                        the graph derived from the Personal Page Rank as specified by idx
                    - source_idx: torch.Tensor of shape (ppr_num_nonzeros)
                        The ids of the training nodes to which the nodes of the ppr_score 
                        are neighbors to
                label: torch.Tensor of shape (batch_size)
                    The labels of the training nodes

        """
        # idx is a list of indices
        key = idx[0]
        if key not in self.cached:
            # shape (batch_size, num_nodes)
            ppr_matrix = self.ppr_matrix[idx]

            # shape (ppr_num_nonzeros)
            source_idx, neighbor_idx = ppr_matrix.nonzero()

            # shape (ppr_num_nonzeros)
            ppr_scores = ppr_matrix.data

            attr_matrix = matrix_to_torch(self.attr_matrix_all[neighbor_idx])
            ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)
            source_idx = torch.tensor(source_idx, dtype=torch.long)

            if self.labels_all is None:
                labels = None
            else:
                labels = self.labels_all[self.indices[idx]]
            self.cached[key] = ((attr_matrix, ppr_scores, source_idx), labels)
        return self.cached[key]
