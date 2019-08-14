import scipy.io as sio
from PIL import Image
import utils as ut
import ann_utils as au
from pycocotools import mask as maskUtils
import os
import numpy as np


class Pascal2012:
    def __init__(self, split, exp_dict, root):
        self.transform_function = ut.bgrNormalize()
        self.collate_fn = ut.collate_fn_0_4
        self.split = split

        self.img_names = []
        self.mask_names = []
        self.cls_names = []

        # train
        train_img_names = ut.load_txt(root + "/ImageSets/Segmentation/train.txt")
        val_img_names = ut.load_txt(root + "/ImageSets/Segmentation/val.txt")

        assert len(train_img_names) == 1464
        assert len(val_img_names) == 1449

        if split == 'train':
            for name in train_img_names:
                name = name.replace("\n", "")
                name_img = os.path.join(root, 'JPEGImages/' + name + '.jpg')
                name_mask =  os.path.join(root, 'SegmentationObject/' +
                                          name  + '.png')
                name_cls =  os.path.join(root, 'SegmentationClass/' + name + '.png')

                self.img_names += [name_img]
                self.mask_names += [name_mask]
                self.cls_names += [name_cls]

            self.img_names.sort()
            self.cls_names.sort()
            self.mask_names.sort()

            self.img_names = np.array(self.img_names)
            self.cls_names = np.array(self.cls_names)
            self.mask_names = np.array(self.mask_names)

        elif split in ['val']:
            for k, name in  enumerate(val_img_names):
                name = name.replace("\n", "")
                name_img = os.path.join(root, 'JPEGImages/' + name + '.jpg')
                name_mask =  os.path.join(root, 'SegmentationObject/' +
                                          name  + '.png')
                name_cls =  os.path.join(root, 'SegmentationClass/' + name + '.png')

                assert os.path.exists(name_img)
                assert os.path.exists(name_mask)
                assert os.path.exists(name_cls)

                self.img_names += [name_img]
                self.mask_names += [name_mask]
                self.cls_names += [name_cls]

        self.n_classes = 21
        self.ignore_index = 255
        self.exp_dict = exp_dict
        
        if split == "val":
            annList_path = "./datasets/annotations/val_gt_annList.json"
            self.annList_path = annList_path

        self.sm_proposal_dict = ut.load_json("./datasets/proposal_dict.json")
        self.prm_point_dict = ut.load_json("./datasets/prm_point_dict.json")

    def __getitem__(self, index, size=None):
        # Image
        img_path = self.img_names[index]

        image_pil = Image.open(img_path).convert('RGB')
        w,h = image_pil.size
        image_id = ut.extract_fname(img_path).split(".")[0]

        #--------------------------------
        # Get annList
        if self.split == "train":
            maskVoid = None
            proposals = self.sm_proposal_dict[image_id]

            # Get the points from the pretrained peak response map
            pointList = self.prm_point_dict[image_id]
            if len(pointList):
                assert pointList[0]["h"] == h
                assert pointList[0]["w"] == w

            # Get AnnList - replace points with proposal of best objectness
            annList = ut.bo_proposal(proposals, image_id, pointList)

        elif self.split == "val":
            pointList = None
            proposals = None

            # groundtruth
            mask_path = self.mask_names[index]
            maskObjects = np.array(load_mask(mask_path))

            cls_path = self.cls_names[index]
            maskClass = np.array(load_mask(cls_path))
            
            maskVoid = maskClass != 255
            maskClass[maskClass==255] = 0
            maskObjects[maskObjects==255] = 0

            annList = au.mask2annList(maskClass, 
                                   maskObjects, 
                                   image_id=image_id,
                                   maskVoid=maskVoid)
        

            maskVoid = maskVoid.astype("uint8")

        targets = au.annList2targets(annList)
        image = self.transform_function(image_pil)

        return {"images":image,
                "pointList":pointList,
                "proposals":proposals,
                "annList":annList,
                "targets":targets,
                "maskVoid":maskVoid,
                "meta":{"index":index, 
                        "image_id":image_id,
                        "shape":(1, 3, h, w)}}

    def __len__(self):
        return len(self.img_names)

        
# ---------------------------------
# Misc
def get_gt_pointList(pointsJSON, image_id, h, w):
    pointList = pointsJSON[image_id]


    tmpList = []
    for i, point in enumerate(pointList):
        if point["y"] > h or point["x"] > w:
            continue

        point["y"] = point["y"]/h
        point["x"] = point["x"]/w
        point["w"] = w
        point["h"] = h
        point["point_id"] = i

        tmpList += [point]

    pointList = tmpList

    return pointList

#------ aux
def load_mask(mask_path):
    if ".mat" in mask_path:
        inst_mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
        inst_mask = Image.fromarray(inst_mask.astype(np.uint8))
    else:
        inst_mask = Image.open(mask_path)

    return inst_mask



# -------------------------------------------------
# helpers
def annList2mask(annList):
    def ann2mask(ann):
        mask =  maskUtils.decode(ann["segmentation"])
        mask[mask==1] = ann["category_id"]
        return mask
    
    mask = None
    for ann in annList:
        if mask is None:
            mask = ann2mask(ann)
        else:
            mask += ann2mask(ann)

    return mask

