from ctypes import Union
from typing import List
from tqdm import tqdm
import numpy as np

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


def ensemble(
    bboxes: List[List[int]], 
    probs: List[List[int]], 
    weights, 
    iou_thr
    ):

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


def start_ensemble(hrnet_detector_result, rcnn_detector_result, weights: List[int], iou_thr: float=0.95):
    """ Парсинг bboxes с двух моделей детекции и начало ансамблирования моделей
    Args:
        hrnet_detector_result: Object - результат работы HRNet модели
        rcnn_detector_result: Object - результат RCNN модели
        weights: List[int] - список весов, которые обозначают значимость работы модели
        iou_thr: float - пороговое значение для отсечения площадей между bboxes
    Return:
        boxes: np.array - массив новых bboxes 
    """
     
    final_bb = []
    final_probs = []
    bboxes_1 = []
    bboxes_2 = []
    prob_1 = []
    prob_2 = []
    prep_res_1 = []
    prep_res_2 = []

    for idx, box in tqdm(enumerate(hrnet_detector_result)):
        prep_res_1.append(box.bbox.to_voc_bbox())
        prep_res_1[idx].append(box.score.value)

    for idx, box in tqdm(enumerate(rcnn_detector_result)):
        prep_res_2.append(box.bbox.to_voc_bbox())
        prep_res_2[idx].append(box.score.value)

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
    boxes, _ = ensemble(final_bb, final_probs, weights, iou_thr)

    return boxes