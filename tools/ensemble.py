from tqdm import tqdm
from utils import load_mmdet_model
from sahi.predict import get_sliced_prediction
import mmcv
import cv2
import numpy as np

def draw_contours(image_array, metadata):
    """ Функция отрисовки контура и подсчета кол-во особей
    Обрабатываем результат работы моделей, извлекая полученный класс животного
    Args:
        image_array: arr - массив-представление изображения
        metadata: json - словарь, содержащий ответ работы моделей
    Return:
        counter_dict: dict - словарь с кол-вом определенных животных
    """
    counter = len(metadata[0])
    for bbox in metadata[0]:
        topLeftCorner = (int(bbox[0]), int(bbox[1]))
        botRightCorner = (int(bbox[2]), int(bbox[3]))

        center_coords = int((botRightCorner[0] + topLeftCorner[0]) / 2), int((botRightCorner[1] + topLeftCorner[1]) / 2)

        cv2.putText(image_array, '*', 
                            center_coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.3, (255, 0, 0),
                            1,
                            1)

    return counter

def soft_nms_float(dets, sc, Nt, thresh: int = 0.4):
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    h = dets[:, 3]
    w = dets[:, 2]
    scores = sc
    areas = h * w

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w_1 = np.maximum(0.0, xx2 - xx1)
        h_1 = np.maximum(0.0, yy2 - yy1)
        inter = w_1 * h_1
        ovr = inter / (areas[i] + areas[pos:] - inter)

        weight = np.ones(ovr.shape)
        weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep

def ensemble(bboxes, probs, weights, iou_thr):
    final_boxes = []
    final_scores = []
    if weights is not None:
        if len(bboxes) != len(weights):
            print('Incorrect number of weights: {}. Must be: {}. Skip it'.format(len(weights), len(bboxes)))
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                probs[i] = (np.array(probs[i]) * weights[i]) / weights.sum()
    bboxes = np.concatenate(bboxes)
    probs = np.concatenate(probs)
    keep = soft_nms_float(bboxes, probs, Nt=iou_thr)
    final_boxes.append(bboxes[keep])
    final_scores.append(probs[keep])
    return final_boxes, final_scores

if __name__ == "__main__":
    hrnet_detector = load_mmdet_model('source/detection_model_rcnn_hr/', 0.85)
    rcnn_detector = load_mmdet_model('source/detection_model_rcnn/', 0.85)
    image = mmcv.imread('test-public-images/photo_2022-05-29 05.40.18.jpeg')

    result_1 = get_sliced_prediction(image, hrnet_detector, slice_height = 1024, slice_width = 1024).object_prediction_list
    result_2 = get_sliced_prediction(image, rcnn_detector, slice_height = 1024, slice_width = 1024).object_prediction_list
    
    final_bb = []
    final_probs = []
    final_labels = []
    bboxes_1 = []
    bboxes_2 = []
    prob_1 = []
    prob_2 = []
    weights = [1, 2]
    iou_thr = 0.95
    prep_res_1 = []
    prep_res_2 = []
    prep_res = []
    for idx, box in tqdm(enumerate(result_1)):
        prep_res_1.append(box.bbox.to_voc_bbox())
        prep_res_1[idx].append(box.score.value)

    for idx, box in tqdm(enumerate(result_2)):
        prep_res_2.append(box.bbox.to_voc_bbox())
        prep_res_2[idx].append(box.score.value)

    # prep_res.append(prep_res_2)
    # prep_res.append(prep_res_1)

    for res in tqdm(prep_res_1):
        bboxes_1.append(res[:-1])
        prob_1.append(res[-1])

    for res in tqdm(prep_res_2):
        bboxes_2.append(res[:-1])
        prob_2.append(res[-1])
    print(f'hrnet - {len(bboxes_1)}')
    print(f'rcnn - {len(bboxes_2)}')
    final_bb.append(bboxes_1)
    final_bb.append(bboxes_2)
    final_probs.append(prob_1)
    final_probs.append(prob_2)
    boxes, scrs = ensemble(final_bb, final_probs, weights, iou_thr)
    count_class = draw_contours(image, boxes)
    print(count_class)
    cv2.imwrite('submission/ensemble.jpg', image)
