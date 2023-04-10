import cv2
import numpy as np
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from tool import infotrans
import numpy as np
import os
import time
import torch

class Nanodet(object):
    def __init__(self, cfg, model_path, logger, device="cpu:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        """
        由可视化函数修改得的信息输出函数

        Outputs:
            bboxes (int): [x,y,w,h]
            confidences (float): 置信度
            class_ids (int): 类别
        """
        time1 = time.time()
        result_img, all_box = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        self.class_names = class_names
        bboxes , confidences , class_ids = infotrans(all_box)
        print("viz time: {:.3f}s".format(time.time() - time1))
        return bboxes , confidences , class_ids

    def draw_bboxes(self, image, bboxes, confidences, class_ids):
        """
        Draw the bounding boxes about detected objects in the image.

        Args:
            image (numpy.ndarray): Image or video frame.
            bboxes (numpy.ndarray): Bounding boxes pixel coordinates as (xmin, ymin, width, height)
            confidences (numpy.ndarray): Detection confidence or detection probability.
            class_ids (numpy.ndarray): Array containing class ids (aka label ids) of each detected object.

        Returns:
            numpy.ndarray: image with the bounding boxes drawn on it.
        """

        for bb, conf, cid in zip(bboxes, confidences, class_ids):
            self.bbox_colors = {key: np.random.randint(0, 255, size=(3,)).tolist() for key in self.class_names.keys()}
            clr = [int(c) for c in self.bbox_colors[cid]]
            cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)
            label = "{}:{:.4f}".format(self.class_names[cid], conf)
            (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(bb[1], label_height)
            cv2.rectangle(image, (bb[0], y_label - label_height), (bb[0] + label_width, y_label + baseLine),
                         (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (bb[0], y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        return image
