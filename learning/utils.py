import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

COLORS1 = [
    (94, 241, 242), (0, 153, 143), (0, 255, 211), (128, 128, 128), (148, 255, 181), (143, 124, 0),
    (157, 204, 0), (194, 0, 136), (255, 164, 5), (255, 168, 187),
    (66, 102, 0), (255, 0, 16), (94, 241, 242), (0, 153, 143), (224, 255, 102),
    (116, 10, 255), (153, 0, 0), (255, 255, 128), (255, 255, 0), (255, 80, 5)
]


def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        mode='max', img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses)+1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[1].plot(np.arange(1, len(step_scores)+1), step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores)+1), eval_scores, color='r', marker='')
    if mode == 'max':
        axes[1].set_ylim(0.5, 1.0)
    else:    # mode == 'min'
        axes[1].set_ylim(0.0, 0.5)
    axes[1].set_ylabel('Error rate')
    axes[1].set_xlabel('Number of epochs')

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)
    plt.close()

def pixelAccuracy(y_pred, y_true, class_idx):
    y_pred = np.argmax(y_pred, axis=0)
    y_true = np.squeeze(y_true, axis=0)

    tp = np.sum((y_pred == y_true)*(y_true == class_idx))
    fp = np.sum((y_pred == class_idx)) - tp
    fn = np.sum((y_true == class_idx)) - tp

    if (tp+fp+fn) == 0:
        return -1.0
    return np.asarray(1.0 * tp / (tp + fp + fn))

def computeIoU(y_pred_batch, y_true_batch):
    class_num = y_pred_batch.shape[1]
    iou = np.zeros(class_num)
    for class_idx in range(class_num):
        pixelAcc = [pixelAccuracy(y_pred_batch[i], y_true_batch[i], class_idx) \
         for i in range(y_true_batch.shape[0])]
        iou[class_idx] = meanIoU(pixelAcc)
    return np.mean(iou[1:])

def meanIoU(acc_list):
    acc_list = [x for x in acc_list if x != -1.0]

    if not acc_list:
        return -1.0
    return np.mean([x for x in acc_list if x != -1.0])

def draw_pixel(y_pred, threshold=None):
    color = COLORS1
    is_batch = len(y_pred.shape) == 4

    if not is_batch:
        y_pred = np.expand_dims(y_pred, axis=0)

    class_num = y_pred.shape[-1]

    if threshold is None:
        mask_pred = np.argmax(mask_pred, axis=-1)
        mask_output = np.zeros(list(mask_pred.shape)+[3])
        for i in range(1, class_num):
            mask_output[np.where(mask_pred==i)] = color[i]
    else:
        mask_output = np.zeros(list(mask_pred.shape[:-1])+[3])
        for i in range(1, class_num):
            mask_output[np.where(mask_pred[...,i] > threshold)] = color[i]

    if is_batch:
        return mask_output
    else:
        return mask_output[0]