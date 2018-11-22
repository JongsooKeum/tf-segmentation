import numpy as np
import os
import glob
import cv2
from datasets.utils import padding, crop_shape

EXT = ['jpg', 'png', 'jpeg']

def read_data(data_dir, image_size, no_label=False):
    """
    Load the data and preprocessing for Segmentation task
    :param data_dir: str, path to the directory to read.
    :image_size: tuple, image size for padding or crop images
    :no_label: bool, whetehr to load labels
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, H, W, num_classes (include background)).
    """
    im_paths = []
    for x in EXT:
        im_paths.extend(glob.glob(os.path.join(data_dir, 'images', '*.{}'.format(x))))
    imgs = []
    labels = []
    for im_path in im_paths:
        # image load
        im_name = os.path.splitext(os.path.basename(im_path))[0]
        im = cv2.imread(im_path)
        im = crop_shape(im, image_size)
        im = padding(im, image_size)
        imgs.append(im)

        if no_label:
            labels.append(0)
            continue

        # mask load
        mask_path = os.path.join(data_dir, 'masks', '{}.png'.format(im_name))
        mask = cv2.imread(mask_path)
        mask = crop_shape(mask, image_size)
        mask = padding(mask, image_size, fill=2)

        label = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)
        label.fill(-1)
        # Pixel annotations 1:Foreground, 2:Background, 3:Unknown
        idx = np.where(mask == 2)
        label[idx[0],idx[1],:] = [1, 0, 0]

        idx = np.where(mask == 1)
        if im_name[0].isupper():
            label[idx[0],idx[1],:] = [0, 1, 0]
        else:
            label[idx[0],idx[1],:] = [0, 0, 1]
        labels.append(label)

    X_set = np.array(imgs, dtype=np.float32)
    y_set = np.array(labels, dtype=np.float32)

    return X_set, y_set

class DataSet(object):

    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C)
        :param labels: np.ndarray, shape: (N, H, W, num_classes (include background)).
        """
        if labels is not None:
            assert images.shape[0] == labels.shape[0],\
                ('Number of examples mismatch, between images and labels')
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels  # NOTE: this can be None, if not given.
        # image/label indices(can be permuted)
        self._indices = np.arange(self._num_examples, dtype=np.uint)
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def sample_batch(self, batch_size, shuffle=True):
        """
        Return sample examples from this dataset.
        :param batch_size: int, size of a sample batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, H, W, num_classes (include background))
        """

        if shuffle:
            indices = np.random.choice(self._num_examples, batch_size)
        else:
            indices = np.arange(batch_size)
        batch_images = self._images[indices]
        if self._labels is not None:
            batch_labels = self._labels[indices]
        else:
            batch_labels = None
        return batch_images, batch_labels

    def next_batch(self, batch_size, shuffle=True):
        """
        Return the next 'batch_size' examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, H, W, num_classes (include background))
        """

        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number
        # of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = self._indices[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]

            images_rest_part = self._images[indices_rest_part]
            images_new_part = self._images[indices_new_part]
            batch_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0)
            if self._labels is not None:
                labels_rest_part = self._labels[indices_rest_part]
                labels_new_part = self._labels[indices_new_part]
                batch_labels = np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index]
            batch_images = self._images[indices]
            if self._labels is not None:
                batch_labels = self._labels[indices]
            else:
                batch_labels = None

        return batch_images, batch_labels
