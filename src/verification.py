# This code is adapted from https://github.com/fdbtrs/ElasticFace/blob/main/eval/verification.py

import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
import torchvision.transforms.functional as tfm
from PIL import Image
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set]
            )
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set]
            )
        unique_far_train, unique_indices = np.unique(far_train, return_index=True)
        unique_thresholds = thresholds[unique_indices]
        if np.max(unique_far_train) >= far_target:
            # Use unique FAR values for interpolation
            f = interpolate.interp1d(unique_far_train, unique_thresholds, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set]
        )

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        pca=pca,
    )
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        1e-3,
        nrof_folds=nrof_folds,
    )
    return tpr, fpr, accuracy, val, val_std, far


# @torch.no_grad()
# def load_bin(path, image_size):
#     try:
#         with open(path, "rb") as f:
#             bins, issame_list = pickle.load(f)  # py2
#     except UnicodeDecodeError as e:
#         with open(path, "rb") as f:
#             bins, issame_list = pickle.load(f, encoding="bytes")  # py3
#     data_list = []
#     for flip in [0, 1]:
#         data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
#         data_list.append(data)
#     for idx in range(len(issame_list) * 2):
#         _bin = bins[idx]
#         img = mx.image.imdecode(_bin)
#         if img.shape[1] != image_size[0]:
#             img = mx.image.resize_short(img, image_size[0])
#         img = nd.transpose(img, axes=(2, 0, 1))
#         for flip in [0, 1]:
#             if flip == 1:
#                 img = mx.ndarray.flip(data=img, axis=2)
#             data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
#         if idx % 1000 == 0:
#             print("loading bin", idx)
#     print(data_list[0].shape)
#     return data_list, issame_list


def load_data_fs(path: str):
    data_list = []
    issame_list = []

    path = Path(path)  # type: Path
    assert path.exists()
    assert (path / "pair_list.txt").exists(), "pair_list.txt not found"
    jpg_files = sorted(list(path.glob("*.jpg")))
    assert len(jpg_files) > 1, "No jpg file found"

    df = pd.read_csv(
        path / "pair_list.txt", sep=" ", header=None, names=["img_1", "img_2", "issame"]
    )
    issame_list = df["issame"].astype(bool).to_list()

    images = []
    images_flipped = []
    print("Reading images from filesytem..")
    for img_fp in jpg_files:
        img = tfm.to_tensor(Image.open(img_fp))
        img_flipped = tfm.hflip(img)
        images.append(img)
        images_flipped.append(img_flipped)

    # torch.stack would throw an error if the image dimensions would not match
    data_list = [torch.stack(images), torch.stack(images_flipped)]
    print(data_list[0].shape)
    print(len(issame_list))

    return data_list, issame_list


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    print("testing verification..")
    device = next(backbone.parameters()).device
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size : bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img.to(device))
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count) :, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print("infer time", time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


# def dumpR(data_set, backbone, batch_size, name="", data_extra=None, label_shape=None):
#     print("dump verification embedding..")
#     data_list = data_set[0]
#     issame_list = data_set[1]
#     embeddings_list = []
#     time_consumed = 0.0
#     for i in range(len(data_list)):
#         data = data_list[i]
#         embeddings = None
#         ba = 0
#         while ba < data.shape[0]:
#             bb = min(ba + batch_size, data.shape[0])
#             count = bb - ba

#             _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
#             time0 = datetime.datetime.now()
#             if data_extra is None:
#                 db = mx.io.DataBatch(data=(_data,), label=(_label,))
#             else:
#                 db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label,))
#             model.forward(db, is_train=False)
#             net_out = model.get_outputs()
#             _embeddings = net_out[0].asnumpy()
#             time_now = datetime.datetime.now()
#             diff = time_now - time0
#             time_consumed += diff.total_seconds()
#             if embeddings is None:
#                 embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
#             embeddings[ba:bb, :] = _embeddings[(batch_size - count) :, :]
#             ba = bb
#         embeddings_list.append(embeddings)
#     embeddings = embeddings_list[0] + embeddings_list[1]
#     embeddings = sklearn.preprocessing.normalize(embeddings)
#     actual_issame = np.asarray(issame_list)
#     outname = os.path.join("temp.bin")
#     with open(outname, "wb") as f:
#         pickle.dump((embeddings, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    path = os.environ["DATASET_DIR"] + "/EvalDatasets/imgdirs/lfw"
    data_set = load_data_fs(path)
