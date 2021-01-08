import logging
import time
import numpy as np
import torch
import gc
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


from pprgo import utils, ppr
from pprgo.pprgo import PPRGo
from pprgo.pytorch_utils import matrix_to_torch


def get_local_logits(model, attr_matrix, batch_size=10000):
    device = next(model.parameters()).device

    nnodes = attr_matrix.shape[0]
    logits = []
    with torch.set_grad_enabled(False):
        for i in range(0, nnodes, batch_size):
            batch_attr = matrix_to_torch(
                attr_matrix[i:i + batch_size]).to(device)
            logits.append(model(batch_attr).to('cpu').numpy())
    logits = np.row_stack(logits)
    return logits


def predict_power_iter(model,
                       adj_matrix,
                       attr_matrix,
                       alpha,
                       nprop=2,
                       inf_fraction=1.0,
                       ppr_normalization='sym',
                       batch_size_logits=10000):
    """
    A more efficient and more accurate prediction method for PPRGo.
    Approximates the full PPR matrix (not just ppr_topk) using a variant of power iteration.
    """
    assert isinstance(
        model, PPRGo), "This prediction method is only ment for PPRGo"

    model.eval()

    start = time.time()
    if inf_fraction < 1.0:
        idx_sub = np.random.choice(adj_matrix.shape[0], int(
            inf_fraction * adj_matrix.shape[0]), replace=False)
        idx_sub.sort()
        attr_sub = attr_matrix[idx_sub]
        logits_sub = get_local_logits(model.mlp, attr_sub, batch_size_logits)
        local_logits = np.zeros(
            [adj_matrix.shape[0], logits_sub.shape[1]], dtype=np.float32)
        local_logits[idx_sub] = logits_sub
    else:
        local_logits = get_local_logits(
            model.mlp, attr_matrix, batch_size_logits)
    time_logits = time.time() - start

    start = time.time()
    row, col = adj_matrix.nonzero()
    logits = local_logits.copy()

    if ppr_normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
        for _ in range(nprop):
            logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (
                deg_sqrt_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'col':
        deg_col = adj_matrix.sum(0).A1
        deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
        for _ in range(nprop):
            logits = (1 - alpha) * (adj_matrix @
                                    (deg_col_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'row':
        deg_row = adj_matrix.sum(1).A1
        deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            logits = deg_row_inv_alpha[:, None] * \
                (adj_matrix @ logits) + alpha * local_logits
    else:
        raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
    predictions = logits.argmax(1)
    time_propagation = time.time() - start

    return predictions, time_logits, time_propagation


def predict_batched(model,
                    dataset_class,
                    adj_matrix,
                    attr_matrix,
                    labels,
                    alpha,
                    eps,
                    topk,
                    ppr_normalization,
                    batch_size=2048):
    """

    """

    model.eval()
    device = next(model.parameters()).device

    try:
        num_nodes = adj_matrix.n_rows
    except AttributeError:
        num_nodes = adj_matrix.shape[0]

    idx = np.arange(num_nodes)

    start = time.time()
    topk = ppr.topk_ppr_matrix(adj_matrix, alpha, eps, idx, topk,
                               normalization=ppr_normalization)
    data_set = dataset_class(attr_matrix_all=attr_matrix,
                             ppr_matrix=topk,
                             indices=idx,
                             labels_all=labels,
                             allow_cache=False)

    data_loader = torch.utils.data.DataLoader(
        dataset=data_set,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(data_set),
            batch_size=batch_size, drop_last=False
        ),
        batch_size=None,
        num_workers=0,
    )
    time_inference_preprocessing = time.time() - start
    start = time.time()
    step = 0
    n_steps = len(data_loader)
    predictions = None

    for xbs, yb in data_loader:
        batch_start = time.time()
        xbs, yb = [xb.to(device)
                   for xb in xbs], yb.to(device)

        with torch.set_grad_enabled(False):
            logits = model(*xbs)
            preds = torch.argmax(logits, dim=1)[:, None]

        if predictions is None:
            predictions = np.row_stack((preds.cpu().numpy()))
        else:
            predictions = np.row_stack((predictions, preds.cpu().numpy()))

        step_percent = step/n_steps * 100
        mib_factor = 1 / 1024 / 1024
        gib_factor = mib_factor / 1024
        gpu_memory = torch.cuda.max_memory_allocated(
        ) * gib_factor if torch.cuda.is_available() else 0
        main_memory = utils.get_max_memory_bytes() * gib_factor
        predictions_memory = predictions.nbytes * mib_factor

        batch_time = time.time() - batch_start
        logging.info(
            f"Inference prediction step {step}/{n_steps}: completed {step_percent:.2f}% "
            f"Memory Usage: {gpu_memory:.2f} GiB GPU, {main_memory:.2f} GiB Main, {predictions_memory:.2f} MiB for predictions"
            f"Batch time is {batch_time:.2f} sec")
        step += 1

    time_inference_prediction = time.time() - start
    return predictions.flatten(), time_inference_preprocessing, time_inference_prediction
