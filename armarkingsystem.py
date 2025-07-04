import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # YOLOv8 segmentation model
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

snapshot = None
reference_keypoints = None
reference_descriptors = None
marked_label = None
toggle_marker = False

def preprocess_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def extract_features(image):
    gray = preprocess_image(image)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def get_largest_in_crosshair(boxes, cross_box):
    best_box = None
    max_overlap_area = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        i_x1 = max(cross_box[0], x1)
        i_y1 = max(cross_box[1], y1)
        i_x2 = min(cross_box[2], x2)
        i_y2 = min(cross_box[3], y2)

        if i_x1 < i_x2 and i_y1 < i_y2:
            overlap_area = (i_x2 - i_x1) * (i_y2 - i_y1)
            if overlap_area > max_overlap_area:
                max_overlap_area = overlap_area
                best_box = box

    return best_box

def match_and_draw(frame, boxes):
    global marked_label, reference_descriptors, reference_keypoints
    for box in boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        roi = frame[y1:y2, x1:x2]
        kp2, des2 = extract_features(roi)

        if des2 is not None and reference_descriptors is not None:
            matches = bf.match(reference_descriptors, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 15:
                label_pos = (x1, y1 - 10)
                cv2.putText(frame, f"MARK ({marked_label})", label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            boxes.append({
                'xyxy': box.xyxy[0].tolist(),
                'label': label
            })

    h, w = frame.shape[:2]
    ch_w, ch_h = 50, 50
    ch_x1, ch_y1 = w // 2 - ch_w // 2, h // 2 - ch_h // 2
    ch_x2, ch_y2 = w // 2 + ch_w // 2, h // 2 + ch_h // 2
    cross_box = (ch_x1, ch_y1, ch_x2, ch_y2)

    if toggle_marker and snapshot is None:
        target = get_largest_in_crosshair(boxes, cross_box)
        if target:
            x1, y1, x2, y2 = map(int, target['xyxy'])
            snapshot = frame[y1:y2, x1:x2]
            reference_keypoints, reference_descriptors = extract_features(snapshot)
            marked_label = target['label']
            print(f"📸 Marked: {marked_label}")
        toggle_marker = False

    if snapshot is not None:
        match_and_draw(frame, boxes)

    # Optional crosshair
    # cv2.rectangle(frame, (ch_x1, ch_y1), (ch_x2, ch_y2), (255, 255, 255), 1)

    cv2.imshow("Helmet View", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('m'):
        snapshot = None
        reference_keypoints = None
        reference_descriptors = None
        marked_label = None
        toggle_marker = True

cap.release()
cv2.destroyAllWindows()
