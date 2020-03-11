import numpy as np
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow(3)


def loss(labels, logits):
    out = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    out = tf.reduce_mean(out)
    return out


def acc(labels, predicts):
    out = tf.keras.metrics.binary_accuracy(labels, predicts, threshold=0.5)
    out = tf.reduce_mean(out)
    return out


def bin_prob_vectors_to_labels(vectors):
    assert len(vectors.shape) == 2 and vectors.shape[1] == 2
    labels = []
    for v in vectors:
        labels.append(np.argmax(v))
    return np.array(labels).astype(np.int8)


poss_counts = (20,50,100,200,400)
round_number = 4


def bin_prob_vectors_to_labels_thr(vectors, threshold=0.5):
    vectors = np.array(vectors)
    assert len(vectors.shape) == 2 and vectors.shape[1] == 2
    labels = []
    for v in vectors:
        if v[1] >= threshold:  # TODO NaNs
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels).astype(int)

def get_TFPN_vectors_thr(predictions,labels,thr):
    assert labels.shape
    try:
        len(labels[0])
        labels = bin_prob_vectors_to_labels(labels)
    except Exception:
        pass
    try:
        len(predictions[0])
        predictions = bin_prob_vectors_to_labels_thr(predictions,thr)
    except Exception:
        predictions = bin_1_to_labels_thr(predictions, thr)
    return get_TFPN_labels(predictions,labels)


def get_TFPN_labels(predictions,labels):
    assert len(labels.shape) == 1
    assert labels.dtype == int or labels.dtype == np.int8
    assert len(predictions.shape) == 1
    assert predictions.dtype == int or predictions.dtype == np.int8

    TP,FP,TN,FN = 0,0,0,0
    for i in range(len(labels)):
        p = predictions[i]
        l = labels[i]
        if (p==1):
            if (l==1):
                TP+=1
            else:
                FP+=1
        else:
            if (l==0):
                TN+=1
            else:
                FN+=1
    return TP,FP,TN,FN


def bin_1_to_labels_thr(vectors, threshold=0.5):
    labels = []
    for v in vectors:
        if v >= threshold:  # TODO NaNs
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels).astype(int)


def metrics_EER(predictions, labels, counts=100):
    if len(labels.shape)==2 and labels.shape[1] == 2:
        labels = bin_prob_vectors_to_labels(labels)
    if counts not in poss_counts:
        counts = 100
    h = round(1/counts, 4)
    FPRs = []
    FNRs = []
    FPR_FNR_diff = 1.0
    EER = 1.0
    EER_steps = [0]
    thr = 0
    thrs = []
    for step in range(counts+1):
        TP, FP, TN, FN = get_TFPN_vectors_thr(predictions, labels, thr)
        FNR = FN/(FN+TP)
        FPR = FP/(FP+TN)
        FNRs.append(FNR)
        FPRs.append(FPR)
        thrs.append(thr)
        thr = round(thr+h, 4)
        eer = abs(FPR-FNR)
        if eer < FPR_FNR_diff:
            FPR_FNR_diff = eer
            EER_steps = [step]
            EER = (FPR+FNR)/2
    thr_EER = round(EER_steps[0]*h, round_number)
    metrics_data = (thrs,FPRs,FNRs)
    return EER, thr_EER, metrics_data