import numpy as np
import cv2 as cv
from motrackers.detectors import Nanodet
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
from nanodet.util import Logger, cfg, load_config, load_model_weight

VIDEO_FILE = "test.avi"
WEIGHTS_PATH = 'weight/LiquidV5.pth'
CONFIG_FILE_PATH = 'config/LiquidDetect416.yml'
CHOSEN_TRACKER = 'SORT'
CONFIDENCE_THRESHOLD = 0.4 # 目标检测的置信度筛选



if CHOSEN_TRACKER == 'CentroidTracker':
    tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
elif CHOSEN_TRACKER == 'CentroidKF_Tracker':
    tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
elif CHOSEN_TRACKER == 'SORT':
    tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
elif CHOSEN_TRACKER == 'IOUTracker':
    tracker = IOUTracker(max_lost=2, iou_threshold=0.5, min_detection_confidence=0.4, max_detection_confidence=0.7,
                         tracker_output_format='mot_challenge')
else:
    print("Please choose one tracker from the above list.")

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
        bboxes,confidences,class_ids,updated_image  = model.visualize(res[0], meta, cfg.class_names, CONFIDENCE_THRESHOLD)
        
        tracks = tracker.update(bboxes, confidences, class_ids)

        updated_image = draw_tracks(updated_image, tracks)

        cv.imshow("image", updated_image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    


main(VIDEO_FILE, detmodel, tracker)