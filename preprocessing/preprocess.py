import numpy as np
import cv2
from PIL import Image

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def get_largest_bbox_cv(image_cv):
    """가장 큰 contour를 객체로 간주하여 bbox 반환. (YOLO 없이 빠르게 사용 가능)"""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        h, w = image_cv.shape[:2]
        return 0, 0, w, h
    
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, x + w, y + h


def preprocess_image(input_pil, target_size=512, object_ratio=0.75):
    """
    PIL → numpy 변환 후:
    1) 객체 bbox 찾기
    2) crop
    3) 중앙 정렬한 정사각형 이미지로 패딩
    4) 객체 크기가 전체 이미지의 object_ratio 되도록 스케일 조절
    5) target_size로 최종 resize
    6) PIL로 다시 반환
    """
    img_cv = pil_to_cv(input_pil)

    # --- Step 1: bbox ---
    x1, y1, x2, y2 = get_largest_bbox_cv(img_cv)
    cropped = img_cv[y1:y2, x1:x2]
    h, w = cropped.shape[:2]

    # --- Step 2: 정사각형 패딩 ---
    side = max(h, w)
    square = np.zeros((side, side, 3), dtype=np.uint8) + 128  # gray padding
    square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = cropped

    # --- Step 3: 객체가 object_ratio 비율이 되도록 스케일 조절 ---
    obj_max = max(h, w)
    target_obj_size = int(target_size * object_ratio)
    scale = target_obj_size / obj_max
    new_side = int(side * scale)
    resized_square = cv2.resize(square, (new_side, new_side))

    # --- Step 4: target_size로 패딩 ---
    final = np.zeros((target_size, target_size, 3), dtype=np.uint8) + 128
    oy = (target_size - new_side) // 2
    ox = (target_size - new_side) // 2
    final[oy:oy+new_side, ox:ox+new_side] = resized_square

    return cv_to_pil(final)
