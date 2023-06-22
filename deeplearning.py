import os
import numpy as np
import cv2

# LOAD YOLO MODEL
from static.extract_text_cnn.lp_recognition import E2E

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/detect_license_yolo/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_detections(img, net):
    # CONVERT IMAGE TO YOLO FORMAT (SQUARE IMAGE)
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_supression(input_image, detections):
    # FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE
    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # clean
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    # NMS
    index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()

    return boxes_np, confidences_np, index


def extract_text(image, bbox):
    x, y, w, h = bbox

    roi = image[y:y + h, x:x + w]
    # roi = rotate_and_crop(roi)
    if 0 in roi.shape:
        return ''
    else:
        model = E2E()
        text = model.predict(roi)
        return text


def rotate_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x, y, w, h = cv2.boundingRect(max_cnt)

    rect = cv2.minAreaRect(max_cnt)
    ((cx, cy), (cw, ch), angle) = rect

    M = cv2.getRotationMatrix2D((cx, cy), angle - 90, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    x, y, w, h = cv2.boundingRect(max_cnt)
    cropped = rotated[y:y + h, x:x + w]
    return cropped

def drawings(image, boxes_np, confidences_np, index):
    # drawings
    text_list = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 30), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        text_list.append(license_text)

    return image, text_list


# predictions
def yolo_predictions(img, net):
    ## step-1: detections
    input_image, detections = get_detections(img, net)
    ## step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    ## step-3: Drawings
    result_img, text = drawings(img, boxes_np, confidences_np, index)
    return result_img, text


def object_detection(path, filename):
    # read image
    image = cv2.imread(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    result_img, text_list = yolo_predictions(image, net)
    cv2.imwrite(os.path.join('static/detect_license_yolo/predict/{}').format(filename), result_img)
    return text_list
