import numpy as np
import os
import xml.etree.ElementTree as ET

import chainer
from chainercv.utils import read_image
import dataset_utils


class OriginalDetectionDataset(chainer.dataset.DatasetMixin):
    """
    Original dataset class for detection using SSD
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Gettingg the label names from dataset_utils.py
        self.label_names = dataset_utils.detection_label_names

        self.img_filenames = []
        self.anno_filenames = []
        # for name in sorted(os.listdir(data_dir)):
        for root, dirs, files in os.walk(data_dir):
            for name in sorted(files):
                # If the file is not an image, ignore the file.
                if os.path.splitext(name)[1] != '.jpg':
                    continue
                img_filename = os.path.join(root, name)
                anno_filename = os.path.splitext(img_filename)[0] + '.xml'
                if not os.path.exists(anno_filename):
                    continue
                self.img_filenames.append(img_filename)
                self.anno_filenames.append(anno_filename)

    def __len__(self):
        return len(self.img_filenames)

    def get_example(self, i):
        img_filename = self.img_filenames[i]
        img_name, jpg = os.path.splitext(img_filename)
        img = read_image(img_filename)
        anno = ET.parse(
            os.path.join(img_name + '.xml'))
        bbox = []
        label = []
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

            name = obj.find('name').text.lower().strip()

            label.append(dataset_utils.detection_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        return img, bbox, label

