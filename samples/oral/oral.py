import os
import sys
import numpy as np
import cv2
import argparse
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import imgaug
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# 模型路径，迁移学习，在已经训练好的模型上重新训练
ORAL_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# 数据配置类
class OralConfig(Config):
    # 数据集名称
    NAME = "oral"
    # GPU使用数量
    # GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    # 总分类 1个background和7个口腔分类
    NUM_CLASSES = 1 + 7
    # epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5


# 数据集类
class OralDataset(utils.Dataset):
    # 数据集加载函数
    # subset: "train" 或者 "val"
    # path: 对应数据集的根路径
    def load_oral(self, subset, path=None):
        # 数据集annotations的路径
        if path == None:
            if subset == "train":
                # anns_json_path = ROOT_DIR + '/oral_dataset/train/annotations.json'
                anns_json_path = ROOT_DIR + '/oral_dataset/annotations.json'
                # 数据集图片的路径
                # img_path = ROOT_DIR + '/oral_dataset/train/JPEGImages'
                img_path = ROOT_DIR + '/oral_dataset/JPEGImages'
            else:
                # anns_json_path = ROOT_DIR + '/oral_dataset/val/annotations.json'
                anns_json_path = ROOT_DIR + '/oral_dataset/annotations.json'
                # 数据集图片的路径
                # img_path = ROOT_DIR + '/oral_dataset/val/JPEGImages'
                img_path = ROOT_DIR + '/oral_dataset/JPEGImages'
        else:
            anns_json_path = path + subset + '/annotations.json'
            # 数据集图片的路径
            img_path = path + subset + '/JPEGImages'

        self.add_class("oral", 1, "teeth_top")
        self.add_class("oral", 2, "teeth_bottom")
        self.add_class("oral", 3, "uvula")
        self.add_class("oral", 4, "tongue")
        self.add_class("oral", 5, "pp_wall")
        self.add_class("oral", 6, "tonsil_right")
        self.add_class("oral", 7, "tonsil_left")
        coco = COCO(anns_json_path)
        class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        # Add classes
        # for i in class_ids:
        #     self.add_class("oral", i, coco.loadCats(i)[0]["name"])
        for i in image_ids:
            self.add_image("oral", image_id=i, width=640, height=480,
                           path=img_path + '/' + str(i) + '.jpg',
                           annotations=coco.loadAnns(coco.getAnnIds(
                               imgIds=[i], catIds=class_ids, iscrowd=None)))
        # return coco

    # mask加载函数，根据输入图片id返回这张图所有的mask
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "oral":
            return super(OralDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "oral.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(OralDataset, self).load_mask(image_id)

    # annotation 转RLE
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    # annotation 转mask
    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


# 主函数
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on Oral dataset')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Oral dataset")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--log', required=False,
                        metavar="./logs",
                        help="Path to weights .h5 file")
    args = parser.parse_args()
    print("Command: ", args.command)
    if args.log is None:
        log_path = ROOT_DIR + "/logs"
    else:
        log_path = args.log

    # train import config
    if args.command == "train":
        config = OralConfig()
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=log_path)
        if args.model is None:
            model_path = ORAL_MODEL_PATH
        else:
            model_path = args.model
        model.load_weights(model_path, by_name=True, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


    else:
        print("Please enter correct command")
        quit()

    # train import dataset
    if args.command == "train":
        dataset_train = OralDataset()
        # TODO 自己添加路径path或者使用默认路径
        dataset_train.load_oral('train')
        dataset_train.prepare()

        dataset_val = OralDataset()
        # TODO 自己添加路径path或者使用默认路径
        dataset_val.load_oral('oral')
        dataset_val.prepare()
        augmentation = imgaug.augmenters.Fliplr(0.5)
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

    else:
        print("Please enter correct command")
        quit()
