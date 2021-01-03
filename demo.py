# %%
import os
import time
import logging
import yaml
import ast
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from torch_sparse import SparseTensor
from pprgo import utils, ppr
from pprgo.pprgo import PPRGo, RobustPPRGo
from pprgo.train import train
from pprgo.predict import predict
from pprgo.dataset import PPRDataset, RobustPPRDataset
from pprgo.pytorch_utils import matrix_to_torch


# %%
model_type = "RobustPPRGO"


# %%
# choose dataset class
if model_type == "RobustPPRGO":
    DatasetClass = RobustPPRDataset
    ModelClass = RobustPPRGo
else:
    DatasetClass = PPRDataset
    ModelClass = PPRGo


# %%
# Set up logging
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s (%(levelname)s): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

# %% [markdown]
# # Download dataset

# %%
#!wget --show-progress -O data/reddit.npz https://ndownloader.figshare.com/files/23742119

# %% [markdown]
# # Load config

# %%
with open('config_demo.yaml', 'r') as c:
    config = yaml.safe_load(c)


# %%
# For strings that yaml doesn't parse (e.g. None)
for key, val in config.items():
    if type(val) is str:
        try:
            config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass


# %%
data_file = config['data_file']           # Path to the .npz data file
# Seed for splitting the dataset into train/val/test
split_seed = config['split_seed']
# Number of training nodes divided by number of classes
ntrain_div_classes = config['ntrain_div_classes']
# Attribute normalization. Not used in the paper
attr_normalization = config['attr_normalization']

alpha = config['alpha']               # PPR teleport probability
# Stopping threshold for ACL's ApproximatePR
eps = config['eps']
topk = config['topk']                # Number of PPR neighbors for each node
# Adjacency matrix normalization for weighting neighbors
ppr_normalization = config['ppr_normalization']

hidden_size = config['hidden_size']         # Size of the MLP's hidden layer
nlayers = config['nlayers']            # Number of MLP layers
# Weight decay used for training the MLP
weight_decay = config['weight_decay']
dropout = config['dropout']             # Dropout used for training

lr = config['lr']                  # Learning rate
# Maximum number of epochs (exact number if no early stopping)
max_epochs = config['max_epochs']
batch_size = config['batch_size']          # Batch size for training
# Multiplier for validation batch size
batch_mult_val = config['batch_mult_val']

# Accuracy is evaluated after every this number of steps
eval_step = config['eval_step']
# Evaluate accuracy on validation set during training
run_val = config['run_val']

early_stop = config['early_stop']          # Use early stopping
patience = config['patience']            # Patience for early stopping

# Number of propagation steps during inference
nprop_inference = config['nprop_inference']
# Fraction of nodes for which local predictions are computed during inference
inf_fraction = config['inf_fraction']

# %% [markdown]
# # Load the data

# %%
start = time.time()
(adj_matrix, attr_matrix, labels,
 train_idx, val_idx, test_idx) = utils.get_data(
    f"{data_file}",
    seed=split_seed,
    ntrain_div_classes=ntrain_div_classes,
    normalize_attr=attr_normalization
)
try:
    d = attr_matrix.n_columns
except AttributeError:
    d = attr_matrix.shape[1]
nc = labels.max() + 1
time_loading = time.time() - start
print(f"Runtime: {time_loading:.2f}s")


# %%
len(train_idx)

# %% [markdown]
# # Preprocessing: Calculate PPR scores

# %%
# compute the ppr vectors for train/val nodes using ACL's ApproximatePR
start = time.time()
topk_train = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, train_idx, topk,
                                 normalization=ppr_normalization)
train_set = DatasetClass(attr_matrix_all=attr_matrix,
                         ppr_matrix=topk_train, indices=train_idx, labels_all=labels)
if run_val:
    topk_val = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, val_idx, topk,
                                   normalization=ppr_normalization)
    val_set = DatasetClass(attr_matrix_all=attr_matrix,
                           ppr_matrix=topk_val, indices=val_idx, labels_all=labels)
else:
    val_set = None
time_preprocessing = time.time() - start
print(f"Runtime: {time_preprocessing:.2f}s")


# %%
len(train_set)


# %%
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    sampler=torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(train_set),
        batch_size=64, drop_last=False
    ),
    batch_size=None,
    num_workers=0,
)


# %%


# %% [markdown]
# # Training: Set up model and train

# %%
start = time.time()
model = ModelClass(d, nc, hidden_size, nlayers, dropout, aggr="mean")
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

nepochs, _, _ = train(
    model=model, train_set=train_set, val_set=val_set,
    lr=lr, weight_decay=weight_decay,
    max_epochs=max_epochs, batch_size=batch_size, batch_mult_val=batch_mult_val,
    eval_step=eval_step, early_stop=early_stop, patience=patience)
time_training = time.time() - start
logging.info('Training done.')
print(f"Runtime: {time_training:.2f}s")

# %% [markdown]
# # Inference (val and test)

# %%
start = time.time()
predictions, time_logits, time_propagation = predict(
    model=model, adj_matrix=adj_matrix, attr_matrix=attr_matrix, alpha=alpha,
    nprop=nprop_inference, inf_fraction=inf_fraction,
    ppr_normalization=ppr_normalization)
time_inference = time.time() - start
print(f"Runtime: {time_inference:.2f}s")

# %% [markdown]
# # Collect and print results

# %%
acc_train = 100 * accuracy_score(labels[train_idx], predictions[train_idx])
acc_val = 100 * accuracy_score(labels[val_idx], predictions[val_idx])
acc_test = 100 * accuracy_score(labels[test_idx], predictions[test_idx])
f1_train = f1_score(labels[train_idx], predictions[train_idx], average='macro')
f1_val = f1_score(labels[val_idx], predictions[val_idx], average='macro')
f1_test = f1_score(labels[test_idx], predictions[test_idx], average='macro')

gpu_memory = torch.cuda.max_memory_allocated()
memory = utils.get_max_memory_bytes()

time_total = time_preprocessing + time_training + time_inference


# %%
print(f'''
Accuracy: Train: {acc_train:.1f}%, val: {acc_val:.1f}%, test: {acc_test:.1f}%
F1 score: Train: {f1_train:.3f}, val: {f1_val:.3f}, test: {f1_test:.3f}

Runtime: Preprocessing: {time_preprocessing:.2f}s, training: {time_training:.2f}s, inference: {time_inference:.2f}s -> total: {time_total:.2f}s
Memory: Main: {memory / 2**30:.2f}GB, GPU: {gpu_memory / 2**30:.3f}GB
''')


# %%


# %%
