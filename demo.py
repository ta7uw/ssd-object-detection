import argparse
import matplotlib.pyplot as plot
import datetime
import os
import chainer
import time

from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox

from dataset_utils import detection_label_names


def main():
    start = time.time()

    chainer.config.train = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model')
    parser.add_argument('image')
    args = parser.parse_args()

    label_names = detection_label_names
    model = SSD512(
        n_fg_class=20,
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)
    bboxes, labels, scores = model.predict([img])
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    bbox, label, score = bboxes[0], labels[0], scores[0]

    vis_bbox(
        img, bbox, label, score, label_names=label_names)

    directory, imgfile = os.path.split(args.image)
    name, jpg = os.path.splitext(imgfile)

    date = datetime.date.today()
    plot.savefig(str(date) + str(name) + '.png')

if __name__ == '__main__':
    main()
