def infotrans(all_box):
    # 用液滴左上和右下边框上的点的坐标近似计算液滴中心点坐标
    bboxes, confidences, class_ids = [], [], []
    for i in range(len(all_box)):
        bboxes.append([all_box[i][1],all_box[i][2],all_box[i][3]-all_box[i][1],all_box[i][4]-all_box[i][2]])
        confidences.append(float(all_box[i][5]))
        class_ids.append(all_box[i][0])
    return bboxes , confidences , class_ids
