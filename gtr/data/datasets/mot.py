import os
import logging
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from .common import load_video_json

logger = logging.getLogger(__name__)


def register_mot_instances(name, metadata, json_file, image_root):
    """
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    DatasetCatalog.register(name, lambda: load_video_json(
        json_file, image_root, name, extra_annotation_keys=['instance_id'],
        map_inst_id=True))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="coco", **metadata
    )

categories = [
    {'id': 1, 'name': 'person'},
]

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS = {
    "mot17_halfval": ("MOT17/trainval/", 
        "MOT17/annotations/val_half_conf0.json"),
    "mot17_halftrain": ("MOT17/trainval/", 
        "MOT17/annotations/train_half_conf0.json"),
    "mot17_fulltrain": ("MOT17/trainval/", 
        "MOT17/annotations/fulltrain_conf0.json"),
    "mot17_test": ("MOT17/test/", 
        "MOT17/annotations/test_conf0.json"),
}

def mot_reg(data_dir, anno_dir):
    logger.info(f'Data root is {data_dir:s} anno root {anno_dir:s}')
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_mot_instances(
            key,
            _get_builtin_metadata(),
            os.path.join(anno_dir, json_file) if "://" not in json_file else json_file,
            os.path.join(data_dir, "datasets", image_root),
        )