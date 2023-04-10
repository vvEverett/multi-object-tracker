import numpy as np
import cv2 as cv
from motrackers.detectors import Nanodet
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
from nanodet.util import Logger, cfg, load_config, load_model_weight

VIDEO_FILE = r"D:\shijue\LiquidDrop\22.avi"
WEIGHTS_PATH = r'D:\shijue\multi-object-tracker\weight\LiquidV4.pth'
CONFIG_FILE_PATH = r'D:\shijue\multi-object-tracker\config\LiquidDetect416.yml'

tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')

# 导入模型文件
local_rank = 0
modelpath = WEIGHTS_PATH
device = "cpu:0"
config = CONFIG_FILE_PATH
logger = Logger(local_rank, use_tensorboard=False)
load_config(cfg, config)
detmodel = Nanodet(cfg, modelpath, logger, device)
logger.log('Press "Esc", "q" or "Q" to exit.')

def main(video_path, model, tracker):

    cap = cv.VideoCapture(video_path)
    while True:
        ok, image = cap.read()

        if not ok:
            print("Cannot read the video feed.")
            break
        
        meta, res = model.inference(image)
        bboxes,confidences,class_ids,updated_image  = model.visualize(res[0], meta, cfg.class_names, 0.43)
        
        tracks = tracker.update(bboxes, confidences, class_ids)

        updated_image = draw_tracks(updated_image, tracks)

        cv.imshow("image", updated_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    


main(VIDEO_FILE, detmodel, tracker)