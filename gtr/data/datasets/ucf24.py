import os
import json
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


def _get_builtin_metadata(json_file):
    with open(json_file,'r') as f:
        annodata = json.load(f)
    categories = annodata['categories']
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS = {
    "ucf24_train": ("/", "ucf24/annotations/train.json"),
    "ucf24_val": ("/", "ucf24/annotations/val.json"),
}

def mot_ucf24(data_dir, anno_dir):
    logger.info(f'Data root is {data_dir:s} anno root {anno_dir:s}')
    things_classes_etc = _get_builtin_metadata(os.path.join(anno_dir,_PREDEFINED_SPLITS['ucf24_val'][1]))
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_mot_instances(
            key,
            things_classes_etc,
            os.path.join(anno_dir, json_file) if "://" not in json_file else json_file,
            data_dir,
        )