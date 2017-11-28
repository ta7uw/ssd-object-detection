"""
Set the object name that you want to detect
"""
detection_label_names = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
    "cocacola",
    "water",
    "oolong tea",
    "orange juice",
    "calpis",

)

voc_semantic_segmentation_label_names = (('background',) +
                                         detection_label_names)

# Add color as well
voc_semantic_segmentation_label_colors = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (0, 0, 256),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
    # add
    (13, 64, 0),
    (0, 19, 0),
    (178, 192, 0),
    (3, 64, 128),
    (128, 64, 0),
    (221, 192, 0),
    (128, 12, 0),
    (0, 24, 128),
)
voc_semantic_segmentation_ignore_label_color = (224, 224, 192)