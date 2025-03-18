#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CRAFT와 EasyOCR을 이용한 기울어진 텍스트 인식 시스템

기능:
1. CRAFT로 텍스트 영역 검출 (기울어진 텍스트 포함)
2. 검출된 영역에 바운딩 박스 표시
3. 기울어진 텍스트를 바로 잡기 위한 affine 변환 적용
4. EasyOCR로 바로잡은 텍스트 인식 및 결과 출력
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from skimage import transform as skimage_transform
import math
import argparse
import shutil

# CRAFT와 EasyOCR 설치 여부 확인 및 임포트
try:
    import easyocr
except ImportError:
    print("EasyOCR이 설치되어 있지 않습니다. 설치를 진행합니다.")
    os.system("pip install easyocr")
    import easyocr

# CRAFT 모듈 경로 설정 - 실제 CRAFT 코드 경로로 수정해주세요
# CRAFT_PATH = "../CRAFT-pytorch"
CRAFT_PATH = "./CRAFT-pytorch"
# CRAFT_PATH = "/project/kehyeong/personal/CRAFT-pytorch"
if not os.path.exists(CRAFT_PATH):
    print(f"CRAFT-pytorch 경로({CRAFT_PATH})가 존재하지 않습니다.")
    print("올바른 경로를 설정하거나 저장소를 클론하세요: git clone https://github.com/clovaai/CRAFT-pytorch.git")
    sys.exit(1)

# CRAFT vgg16_bn.py 파일 수정하여 model_urls 문제 해결
vgg16_bn_path = os.path.join(CRAFT_PATH, 'basenet', 'vgg16_bn.py')
if os.path.exists(vgg16_bn_path):
    # 백업 파일 생성
    backup_path = vgg16_bn_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(vgg16_bn_path, backup_path)
        print(f"원본 파일 백업 생성: {backup_path}")
    
    # 파일 내용 읽기
    with open(vgg16_bn_path, 'r') as f:
        content = f.read()
    
    # model_urls 관련 부분 수정
    if 'from torchvision.models.vgg import model_urls' in content:
        # 최신 torchvision에 맞게 수정
        updated_content = content.replace(
            'from torchvision.models.vgg import model_urls',
            '# torchvision 최신 버전 호환을 위한 model_urls 정의\nmodel_urls = {\n    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",\n}'
        )
        
        # 수정된 내용 저장
        with open(vgg16_bn_path, 'w') as f:
            f.write(updated_content)
        print("vgg16_bn.py 파일이 성공적으로 업데이트되었습니다.")

# CRAFT 모듈 임포트
sys.path.append(CRAFT_PATH)
try:
    from craft import CRAFT
    from craft_utils import adjustResultCoordinates, getDetBoxes
    from imgproc import resize_aspect_ratio, normalizeMeanVariance
    from file_utils import saveResult
    print("CRAFT 모듈 로드 성공")
except ImportError as e:
    print(f"CRAFT 모듈 임포트 실패: {e}")
    print("CRAFT-pytorch 저장소가 올바르게 설치되어 있는지 확인하세요.")
    sys.exit(1)

def load_craft_model(model_path):
    """ CRAFT 텍스트 감지 모델 로드 """
    # CUDA 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    
    # CRAFT 모델 초기화
    net = CRAFT()
    
    print(f'모델 파일 로드: {model_path}')
    try:
        # 모델 상태 사전 로드
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        # DataParallel로 저장된 모델 처리 (module. 접두사 제거)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            # module. 접두사 제거
            if name.startswith('module.'):
                name = name[7:]  # 'module.' 부분 제거
            new_state_dict[name] = v
        
        # 수정된 state_dict 로드
        net.load_state_dict(new_state_dict)
        
        if torch.cuda.is_available():
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        
        # 평가 모드로 전환
        net.eval()
        print("모델 로드 성공")
        return net
    
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("모델 파일이 올바른지 확인하세요.")
        sys.exit(1)

def detect_text_regions(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4, 
                        canvas_size=1280, mag_ratio=1.5, poly=False):
    """ CRAFT를 사용하여 이미지에서 텍스트 영역 검출 """
    # 이미지 전처리
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 
                                                                 canvas_size, 
                                                                 interpolation=cv2.INTER_LINEAR, 
                                                                 mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    
    # 정규화
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
    
    # CUDA 사용 가능 여부에 따라 처리
    if torch.cuda.is_available():
        x = x.cuda()
    
    # 모델 추론
    with torch.no_grad():
        y, feature = net(x)
    
    # 결과 후처리
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # 텍스트 영역 박스 검출
    # poly는 불리언 값이므로 그대로 전달
    use_polygon = bool(poly)  # 확실히 불리언으로 변환
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, use_polygon)
    
    # 좌표 조정 - 안전한 방식으로 처리
    ratio_net = 2  # CRAFT 네트워크 내부 비율
    
    # 박스 좌표 조정
    for k in range(len(boxes)):
        boxes[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    
    # 다각형 좌표 조정 (None이 아닌 경우만)
    adjusted_polys = []
    for k in range(len(polys)):
        if polys[k] is not None:
            poly = polys[k].copy()  # 원본 보존
            poly *= (ratio_w * ratio_net, ratio_h * ratio_net)
            adjusted_polys.append(poly)
        else:
            adjusted_polys.append(None)
    
    # 다각형을 사용하지 않는 경우 직사각형 박스로 변환
    if not use_polygon:  # 불리언 값 사용
        polys = boxes
    else:
        polys = adjusted_polys
    
    return boxes, polys, img_resized

def preprocess_image(image, max_size=512, apply_resize=True):
    """
    이미지 전처리 함수 - 텍스트 인식 최적화
    
    다음 단계를 적용:
    1. 이미지 크기 조정 (선택적)
    2. 그레이스케일 변환
    3. 노이즈 제거 (가우시안 블러)
    4. 적응형 이진화
    5. 모폴로지 연산 (침식 및 팽창)
    6. 엣지 강화 및 샤프닝
    7. 대비 향상 (CLAHE)
    
    Args:
        image: 입력 이미지
        max_size: 최대 크기 (픽셀 단위)
        apply_resize: 크기 조정 적용 여부
    
    Returns:
        dict: 다양한 전처리 결과를 포함하는 딕셔너리
    """
    # 원본 이미지 보존
    original_img = image.copy()
    processed_img = image.copy()
    
    # 1. 이미지 크기 조정 (큰 이미지의 경우)
    if apply_resize:
        h, w = processed_img.shape[:2]
        # 가장 큰 변이 max_size보다 큰 경우에만 리사이징
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            processed_img = cv2.resize(processed_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  - 이미지 크기 조정: {w}x{h} -> {new_w}x{new_h}")
    
    # 처리 결과 저장
    results = {'original': original_img, 'resized': processed_img}
    
    # 2. 그레이스케일 변환 
    if len(processed_img.shape) == 3:
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_img.copy()
    results['gray'] = gray
    
    # 3. 노이즈 제거 (블러링)
    # 가우시안 블러로 노이즈 감소
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    results['blurred'] = blurred
    
    # 추가 블러링 - 모아레 패턴 제거
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    results['bilateral'] = bilateral
    
    # 중간 노이즈 제거 블러
    median = cv2.medianBlur(gray, 3)
    results['median'] = median
    
    # 4. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(median)
    results['clahe'] = clahe_img
    
    # 5. 적응형 이진화 (Adaptive Thresholding)
    # 작은 영역 단위로 임계값을 다르게 적용하여 조명 불균일 문제 해결
    binary = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    results['binary'] = binary
    
    # 글로벌 이진화 (Otsu's method)
    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results['otsu'] = otsu
    
    # 6. 모폴로지 연산 - 침식 후 팽창 (Opening)
    # 작은 노이즈 제거
    kernel_small = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    # 7. 모폴로지 연산 - 팽창 후 침식 (Closing)
    # 글자의 끊김 방지
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    results['morphology'] = closing
    
    # 8. 엣지 검출 및 강화
    edges = cv2.Canny(clahe_img, 100, 200)
    results['edges'] = edges
    
    # 9. 샤프닝
    kernel_sharpen = np.array([[-1, -1, -1], 
                              [-1, 9, -1],
                              [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
    results['sharpened'] = sharpened
    
    # 10. 최종 처리 이미지 (이진화 + 모폴로지 + 선명화)
    final_img = cv2.bitwise_and(closing, sharpened)
    results['final'] = final_img
    
    return results

def resize_image_with_aspect_ratio(image, max_size=512):
    """
    이미지의 비율을 유지하면서 크기를 조정합니다.
    가장 큰 차원이 max_size가 되도록 조정합니다.
    
    Args:
        image: 입력 이미지
        max_size: 최대 크기 (픽셀 단위)
    
    Returns:
        리사이징된 이미지
    """
    h, w = image.shape[:2]
    
    # 이미 충분히 작은 경우 그대로 반환
    if max(h, w) <= max_size:
        return image
    
    # 비율 계산
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 리사이징
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def correct_skew(img, box):
    """ 기울어진 텍스트 영역을 affine 변환으로 보정 """
    try:
        # 바운딩 박스를 numpy 배열로 변환
        box_np = np.array(box, dtype=np.float32)
        
        # 최소 영역 사각형 추출
        rect = cv2.minAreaRect(box_np)
        
        # 디버깅 정보 출력
        center, (width, height), angle = rect
        print(f"  - 사각형 정보: 중심={center}, 크기=({width:.1f}, {height:.1f}), 각도={angle:.2f}°")
        
        # 회전된 영역만 추출 후 0도로 바로잡기
        # 각도 조정 (OpenCV의 minAreaRect는 [-90, 0) 범위로 각도 제공)
        # 영어 텍스트는 항상 가로로 읽어야 하므로 수평에 가깝게 조정
        
        # 가로 텍스트 감지 기준
        is_horizontal = abs(angle) < 45 or abs(angle) > 45
        
        if width < height:
            # 세로로 긴 텍스트 (예: 'why')
            # 이 경우 minAreaRect는 90도 각도를 반환하기 때문에 0도로 조정해야 함
            angle = 0  # 회전 없이 그대로 사용
            print(f"  - 세로로 긴 텍스트 감지: 회전 각도 조정 (0도)")
        elif angle < -45:
            # -45도 이하의 각도는 90도를 더해 양수 각도로 변환
            angle = 90 + angle
            print(f"  - 각도 조정: {angle:.2f}°")
        
        # 원본 이미지 전체에 대한 회전 매트릭스 생성
        M = cv2.getRotationMatrix2D(center, angle, 1)
        
        # 회전된 이미지의 크기 계산
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        # 새 크기 계산: 회전 후에도 모든 내용이 포함되도록
        new_w = int((height * sin) + (width * cos)) + 20
        new_h = int((height * cos) + (width * sin)) + 20
        
        # 회전 중심이 새 이미지 중앙에 오도록 조정
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 회전 적용
        rotated = cv2.warpAffine(img.copy(), M, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        # 박스 포인트를 회전된 좌표계로 변환
        box_points = cv2.boxPoints(rect)
        rotated_points = np.zeros_like(box_points)
        
        for i, point in enumerate(box_points):
            # 원래 좌표에 회전 변환 적용
            px = M[0, 0] * point[0] + M[0, 1] * point[1] + M[0, 2]
            py = M[1, 0] * point[0] + M[1, 1] * point[1] + M[1, 2]
            rotated_points[i] = [px, py]
        
        # 회전된 좌표에서 경계 계산
        x_min = max(0, int(np.min(rotated_points[:, 0])))
        y_min = max(0, int(np.min(rotated_points[:, 1])))
        x_max = min(rotated.shape[1], int(np.max(rotated_points[:, 0])))
        y_max = min(rotated.shape[0], int(np.max(rotated_points[:, 1])))
        
        # 여백 추가
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(rotated.shape[1], x_max + margin)
        y_max = min(rotated.shape[0], y_max + margin)
        
        # 회전된 이미지에서 텍스트 영역 추출
        cropped = rotated[y_min:y_max, x_min:x_max].copy()
        
        if cropped.size == 0:
            print("  - 추출 영역이 비어있습니다 - 원본 반환")
            return img, angle
        
        return cropped, angle
    
    except Exception as e:
        print(f"텍스트 영역 보정 오류: {str(e)} - 원본 반환")
        
        # 오류 발생 시 바운딩 박스만 추출
        try:
            # 단순히 바운딩 박스 좌표 추출
            x_min = max(0, int(np.min(box[:, 0])) - 5)
            y_min = max(0, int(np.min(box[:, 1])) - 5)
            x_max = min(img.shape[1], int(np.max(box[:, 0])) + 5)
            y_max = min(img.shape[0], int(np.max(box[:, 1])) + 5)
            
            # 단순히 영역만 추출
            cropped = img[y_min:y_max, x_min:x_max].copy()
            if cropped.size > 0:
                return cropped, 0.0
        except:
            pass
        
        return img, 0.0

def visualize_results(image, boxes, texts=None, output_path=None):
    """ 검출 결과 시각화 """
    img_draw = image.copy()
    
    # 색상 설정
    box_color = (0, 0, 255)  # 빨간색 (BGR)
    text_bg_color = (0, 0, 0)  # 검은색
    text_color = (255, 255, 255)  # 흰색
    
    # 각 박스 처리
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1, 2))
        
        # 다각형 그리기
        cv2.polylines(img_draw, [poly], True, box_color, 2)
        
        # 박스 번호 표시
        center_x = int(np.mean([p[0] for p in poly]))
        center_y = int(np.mean([p[1] for p in poly]))
        
        # 번호 배경
        cv2.circle(img_draw, (center_x, center_y), 15, box_color, -1)
        
        # 번호 텍스트
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        cv2.putText(img_draw, str(i+1), (center_x-5, center_y+5), 
                  font, font_scale, (255, 255, 255), thickness)
        
        # 인식된 텍스트가 있으면 표시
        if texts is not None and i < len(texts) and texts[i]:
            # 텍스트 배치 위치 계산
            min_x = min(p[0] for p in poly)
            min_y = min(p[1] for p in poly) - 5  # 텍스트가 박스 위에 오도록
            
            # 표시할 텍스트
            text = texts[i]
            
            # 텍스트 크기 측정
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 텍스트 배경 그리기 (가독성 향상)
            cv2.rectangle(img_draw, 
                         (min_x, min_y - text_height - 5), 
                         (min_x + text_width, min_y + 5), 
                         text_bg_color, -1)
            
            # 텍스트 추가
            cv2.putText(img_draw, text, (min_x, min_y), 
                       font, font_scale, text_color, thickness)
    
    # 결과 이미지 저장
    if output_path is not None:
        cv2.imwrite(output_path, img_draw)
        print(f"결과 이미지 저장: {output_path}")
    
    # matplotlib으로 이미지 표시
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    return img_draw

def process_image(image_path, craft_net, reader, 
                 text_threshold=0.7, link_threshold=0.4, 
                 low_text=0.4, canvas_size=1280, 
                 mag_ratio=1.5, poly=True, 
                 output_dir='output',
                 paragraph=False, detail=1, allowlist=None,
                 direct_img=False, return_coords=False, return_confidence=False):
    """
    이미지 처리 함수: 텍스트 영역 감지, 기울기 보정, OCR 수행
    
    Args:
        image_path (str or numpy.ndarray): 이미지 파일 경로 또는 이미지 객체
        craft_net: CRAFT 모델
        reader: EasyOCR reader 객체
        text_threshold (float): 텍스트 감지 임계값
        link_threshold (float): 텍스트 링크 임계값
        low_text (float): 낮은 신뢰도 텍스트 임계값
        canvas_size (int): 처리 캔버스 크기
        mag_ratio (float): 이미지 확대 비율
        poly (bool): 다각형 형태로 경계 감지 여부
        output_dir (str): 결과 저장 디렉토리
        paragraph (bool): 단락 구성 여부
        detail (int): 상세 수준 (1: 일반, 2: 세부)
        allowlist (str): 허용 문자 목록
        direct_img (bool): image_path가 이미지 객체인지 여부
        return_coords (bool): 좌표 정보도 반환할지 여부
        return_confidence (bool): 신뢰도 정보도 반환할지 여부
    
    Returns:
        return_coords=False, return_confidence=False:
            튜플: (처리된 이미지, 인식된 텍스트 목록, 텍스트 영역 이미지 목록)
        return_coords=True, return_confidence=False:
            튜플: (처리된 이미지, 인식된 텍스트 목록, 텍스트 영역 이미지 목록, 박스 좌표 목록, 다각형 좌표 목록)
        return_coords=False, return_confidence=True:
            튜플: (처리된 이미지, 인식된 텍스트 목록, 텍스트 영역 이미지 목록, 신뢰도 목록)
        return_coords=True, return_confidence=True:
            튜플: (처리된 이미지, 인식된 텍스트 목록, 텍스트 영역 이미지 목록, 박스 좌표 목록, 다각형 좌표 목록, 신뢰도 목록)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드 (직접 객체 또는 파일 경로)
    if direct_img:
        print(f"이미지 객체 직접 처리")
        img = image_path  # 이미지 객체 직접 사용
    else:
        print(f"이미지 파일 처리: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return None, [], []
    
    img_original = img.copy()
    img_original_for_detection = img.copy()
    
    # 원본 이미지 크기 출력
    print(f"원본 이미지 크기: {img.shape[1]}x{img.shape[0]}")
    
    # 1. 이미지 리사이징 (최대 크기 제한)
    resized_img = resize_image_with_aspect_ratio(img, max_size=1024)
    img = resized_img.copy()
    print(f"CRAFT 처리용 이미지 크기: {img.shape[1]}x{img.shape[0]}")
    
    # 2. 텍스트 영역 감지
    bboxes_original, polys_original, img_original_resized = detect_text_regions(
        craft_net, 
        img, 
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
        poly=poly
    )
    
    # 이미지 전처리 적용
    print("이미지 전처리 적용 중...")
    preprocessed_imgs = preprocess_image(img, max_size=512, apply_resize=True)
    
    # 전처리된 이미지 저장
    for img_type, processed_img in preprocessed_imgs.items():
        if img_type != 'original' and img_type != 'resized':
            # 컬러 변환 필요한 경우 처리
            if len(processed_img.shape) == 2:  # 그레이스케일 이미지
                if img_type in ['edges']:  # 에지는 그대로 저장
                    save_img = processed_img
                else:
                    save_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            else:
                save_img = processed_img
                
            preproc_path = os.path.join(output_dir, f"preproc_{img_type}.jpg")
            cv2.imwrite(preproc_path, save_img)
            print(f"  - {img_type} 이미지 저장: {preproc_path}")
    
    # 이미지 처리 과정 시각화
    visualize_preprocessing_steps(preprocessed_imgs, output_dir)
    
    # 원본 이미지와 전처리된 이미지로 텍스트 영역 검출 시도
    detection_results = []
    
    # 1. 원본 이미지로 검출
    print("원본 이미지로 텍스트 영역 검출 중...")
    orig_boxes, orig_polys, orig_img_resized = detect_text_regions(
        craft_net, img, 
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
        poly=poly
    )
    detection_results.append(('original', orig_boxes, orig_polys, img))
    
    # 2. 이진화 이미지로 검출
    binary_img = preprocessed_imgs['binary']
    binary_img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    print("이진화 이미지로 텍스트 영역 검출 중...")
    binary_boxes, binary_polys, binary_img_resized = detect_text_regions(
        craft_net, binary_img_rgb, 
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
        poly=poly
    )
    detection_results.append(('binary', binary_boxes, binary_polys, binary_img_rgb))
    
    # 3. CLAHE 이미지로 검출
    clahe_img = preprocessed_imgs['clahe']
    clahe_img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    print("CLAHE 이미지로 텍스트 영역 검출 중...")
    clahe_boxes, clahe_polys, clahe_img_resized = detect_text_regions(
        craft_net, clahe_img_rgb, 
        text_threshold=text_threshold,
        link_threshold=link_threshold,
        low_text=low_text,
        canvas_size=canvas_size,
        mag_ratio=mag_ratio,
        poly=poly
    )
    detection_results.append(('clahe', clahe_boxes, clahe_polys, clahe_img_rgb))
    
    # 최적의 결과 선택 (더 많은 텍스트 영역이 검출된 결과 사용)
    best_result = max(detection_results, key=lambda x: len(x[1]))
    result_type, boxes, polys, detection_img = best_result
    
    print(f"최적 검출 결과 선택: {result_type} (검출 영역 수: {len(boxes)})")
    
    # 감지 결과 시각화
    img_detection = visualize_results(img.copy(), boxes, None, 
                                    output_path=os.path.join(output_dir, "detection.jpg"))
    
    if len(boxes) == 0:
        print("텍스트 영역이 감지되지 않았습니다.")
        return img_detection, [], []
    
    print(f"검출된 텍스트 영역 수: {len(boxes)}")
    print(f"결과 이미지 저장: {os.path.join(output_dir, 'detection.jpg')}")
    
    # 각 텍스트 영역 처리
    recognized_texts = []
    all_detected_texts = []  # 각 영역에서 인식된 모든 텍스트 저장
    corrected_images = []
    confidences = []  # 각 영역의 인식 신뢰도
    
    for i, box in enumerate(boxes):
        print(f"영역 {i+1} 처리 중...")
        
        # 이제 기울기 보정 함수를 통해 텍스트 영역 추출
        try:
            # 바운딩 박스를 사용하여 텍스트 영역 추출 및 회전
            cropped_img, angle = correct_skew(img, box)  # 원본 이미지에서 추출
            print(f"  - 텍스트 영역 추출 완료 (각도: {angle:.2f}°)")
            
            # 추출된 이미지 저장
            crop_path = os.path.join(output_dir, f"crop_{i+1}.jpg")
            cv2.imwrite(crop_path, cropped_img)
            corrected_images.append(cropped_img)
            
            # 추출된 이미지에 추가 전처리 적용 (리사이징 없이)
            crop_preprocessed = preprocess_image(cropped_img, apply_resize=False)
            
            # 다양한 전처리 버전으로 OCR 시도 (더 나은 결과 선택)
            ocr_results = []
            
            # 1. 원본으로 OCR
            orig_results = reader.readtext(cropped_img)
            if orig_results:
                ocr_results.append(('original', orig_results))
            
            # 2. 이진화로 OCR
            binary_crop = crop_preprocessed['binary']
            binary_results = reader.readtext(binary_crop)
            if binary_results:
                ocr_results.append(('binary', binary_results))
            
            # 3. 전처리 완료 이미지로 OCR
            final_crop = crop_preprocessed['final']
            final_results = reader.readtext(final_crop)
            if final_results:
                ocr_results.append(('final', final_results))
                
            # 4. 샤프닝 이미지로 OCR
            sharpened = crop_preprocessed['sharpened']
            sharp_results = reader.readtext(sharpened)
            if sharp_results:
                ocr_results.append(('sharpened', sharp_results))
            
            # 5. CLAHE 적용 이미지로 OCR
            clahe_img = crop_preprocessed['clahe']
            clahe_results = reader.readtext(clahe_img)
            if clahe_results:
                ocr_results.append(('clahe', clahe_results))
            
            # 이미지 전처리 결과 저장 (각 영역별)
            process_types = ['binary', 'clahe', 'sharpened', 'final']
            for p_type in process_types:
                if p_type in crop_preprocessed:
                    proc_img = crop_preprocessed[p_type]
                    if len(proc_img.shape) == 2:  # 그레이스케일
                        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(os.path.join(output_dir, f"crop_{i+1}_{p_type}.jpg"), proc_img)
            
            # 가장 좋은 OCR 결과 선택
            if ocr_results:
                # 먼저 결과가 있는 것만 필터링
                valid_results = [(t, r) for t, r in ocr_results if r]
                
                if valid_results:
                    # 결과가 하나라도 있는 경우
                    # 각 전처리 방식 별로 인식된 텍스트의 평균 신뢰도 계산
                    confidence_scores = []
                    for t, r in valid_results:
                        confidence = sum(res[2] for res in r) / len(r)
                        confidence_scores.append((t, r, confidence))
                    
                    # 평균 신뢰도가 가장 높은 결과 선택
                    best_ocr = max(confidence_scores, key=lambda x: x[2])
                    preproc_type, results, confidence = best_ocr
                    print(f"  - 최적 OCR 결과 선택: {preproc_type} (평균 신뢰도: {confidence:.2f})")
                else:
                    # 결과가 없는 경우
                    results = []
                    confidence = 0.0
            else:
                results = []
                confidence = 0.0
            
            # 결과 처리
            if results and len(results) > 0:
                # 모든 텍스트 결과를 저장
                area_texts = []
                for bbox, text, conf in results:
                    area_texts.append((text, conf))
                    print(f"  - 인식 결과: '{text}' (신뢰도: {conf:.2f})")
                
                # 현재 영역의 모든 텍스트 저장
                all_detected_texts.append(area_texts)
                
                # 가장 높은 신뢰도의 텍스트 선택
                best_text_with_conf = sorted(area_texts, key=lambda x: x[1], reverse=True)[0]
                best_text, best_conf = best_text_with_conf
                recognized_texts.append(best_text)
                confidences.append(best_conf)  # 최종 선택된 텍스트의 신뢰도 저장
            else:
                print("  - 텍스트를 인식할 수 없습니다.")
                recognized_texts.append("")
                all_detected_texts.append([])
                confidences.append(0.0)  # 결과가 없는 경우 신뢰도 0
        
        except Exception as e:
            print(f"  - 영역 처리 오류: {str(e)}")
            recognized_texts.append("")
            all_detected_texts.append([])
            confidences.append(0.0)  # 오류 발생 시 신뢰도 0
    
    # 텍스트 결과 저장
    with open(os.path.join(output_dir, "recognized_texts.txt"), "w", encoding="utf-8") as f:
        # 최종 선택된 텍스트 결과 저장
        f.write("=== 최종 선택된 텍스트 ===\n")
        for i, text in enumerate(recognized_texts):
            if text:  # 빈 결과는 건너뜀
                f.write(f"영역 {i+1}: {text}\n")
        
        # 각 영역에서 인식된 모든 텍스트 저장
        f.write("\n=== 각 영역에서 인식된 모든 텍스트 ===\n")
        for i, area_texts in enumerate(all_detected_texts):
            f.write(f"영역 {i+1}:\n")
            for text, confidence in area_texts:
                f.write(f"  - '{text}' (신뢰도: {confidence:.2f})\n")
        
        # 모든 영역의 텍스트를 조합한 최종 결과
        f.write("\n=== 전체 텍스트 ===\n")
        full_text = " ".join([t for t in recognized_texts if t])
        f.write(full_text)
    
    # 최종 결과 시각화 (인식 텍스트 포함)
    img_result = visualize_results(img.copy(), boxes, recognized_texts, 
                                 output_path=os.path.join(output_dir, "result.jpg"))
    
    print(f"처리 완료. 결과는 {output_dir} 폴더에 저장되었습니다.")
    
    # 반환 값 결정
    if return_coords and return_confidence:
        return img_result, recognized_texts, corrected_images, boxes, polys, confidences
    elif return_coords:
        return img_result, recognized_texts, corrected_images, boxes, polys
    elif return_confidence:
        return img_result, recognized_texts, corrected_images, confidences
    else:
        return img_result, recognized_texts, corrected_images

def visualize_preprocessing_steps(preprocessed_imgs, output_dir):
    """
    이미지 전처리 단계를 시각화하여 비교 이미지 생성
    
    Args:
        preprocessed_imgs: 전처리된 이미지 딕셔너리
        output_dir: 결과 저장 디렉토리
    """
    # 비교할 전처리 단계 선택
    steps_to_compare = ['gray', 'blurred', 'bilateral', 'clahe', 'binary', 'sharpened', 'final']
    
    # 유효한 단계만 선택
    valid_steps = [step for step in steps_to_compare if step in preprocessed_imgs]
    
    if len(valid_steps) == 0:
        return
    
    # 3x3 그리드 생성 
    rows = (len(valid_steps) + 2) // 3  # +2는 올림을 위한 처리
    cols = min(3, len(valid_steps))
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, step in enumerate(valid_steps):
        plt.subplot(rows, cols, i + 1)
        
        # 그레이스케일 이미지 처리
        if len(preprocessed_imgs[step].shape) == 2:
            plt.imshow(preprocessed_imgs[step], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(preprocessed_imgs[step], cv2.COLOR_BGR2RGB))
        
        plt.title(f"{step}")
        plt.axis('off')
    
    # 이미지 저장
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "preprocessing_steps.jpg"))
    plt.close()
    
    print(f"전처리 단계 비교 이미지 저장: {os.path.join(output_dir, 'preprocessing_steps.jpg')}")

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='CRAFT + EasyOCR을 이용한 기울어진 텍스트 인식')
    parser.add_argument('--image', '-i', required=True, help='입력 이미지 경로')
    parser.add_argument('--craft_model', '-m', default=os.path.join(CRAFT_PATH, 'weights', 'craft_mlt_25k.pth'), 
                        help='CRAFT 모델 경로')
    parser.add_argument('--output_dir', '-o', default='output', help='결과 저장 경로')
    parser.add_argument('--text_threshold', '-t', type=float, default=0.7, help='텍스트 감지 임계값')
    parser.add_argument('--link_threshold', '-l', type=float, default=0.4, help='링크 감지 임계값')
    parser.add_argument('--low_text', type=float, default=0.4, help='낮은 텍스트 감지 임계값')
    parser.add_argument('--lang', nargs='+', default=['ko', 'en', 'en_numeric'], 
                       help='인식할 언어 목록 (영어: en, 한국어: ko, 숫자: en_numeric)')
    parser.add_argument('--paragraph', action='store_true', help='EasyOCR 문단 모드 사용')
    parser.add_argument('--allowlist', type=str, default=None, help='EasyOCR 허용 문자 목록')
    args = parser.parse_args()
    
    # 모델 경로 확인
    if not os.path.exists(args.craft_model):
        print(f"CRAFT 모델 파일이 존재하지 않습니다: {args.craft_model}")
        print("모델 다운로드 링크: https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ")
        sys.exit(1)
    
    # 이미지 경로 확인
    if not os.path.exists(args.image):
        print(f"입력 이미지가 존재하지 않습니다: {args.image}")
        sys.exit(1)
    
    # CRAFT 모델 로드
    craft_net = load_craft_model(args.craft_model)
    
    # 언어 목록 정리 및 숫자 인식을 위한 설정
    langs = [lang for lang in args.lang if lang != 'en_numeric']
    
    # 숫자 인식을 위한 설정
    num_allowlist = None
    if 'en_numeric' in args.lang:
        print("숫자 인식 활성화")
        # 기본 숫자 및 기호 문자 목록
        if args.allowlist:
            num_allowlist = args.allowlist
        else:
            num_allowlist = '0123456789.,+-*/=%()[]{}><'
    
    # EasyOCR 리더 초기화
    print(f"EasyOCR 초기화 중 (언어: {', '.join(langs)})...")
    
    # EasyOCR 버전 확인 및 매개변수 설정
    try:
        # 현재 버전 확인
        import pkg_resources
        easyocr_version = pkg_resources.get_distribution("easyocr").version
        print(f"EasyOCR 버전: {easyocr_version}")
        
        # 추가 매개변수 설정
        reader_params = {
            'lang_list': langs,
            'gpu': torch.cuda.is_available(),
            'quantize': False,  # 정확도 우선
            'verbose': False
        }
        
        # 숫자 목록이 있으면 필터 적용을 시도합니다 (버전별 다른 매개변수명 시도)
        if num_allowlist:
            # 새로운 버전에서는 'allowlist'
            try:
                reader = easyocr.Reader(**reader_params, allowlist=num_allowlist)
                print("allowlist 매개변수 적용됨")
            except TypeError:
                # 이전 버전에서는 'char_whitelist'
                try:
                    reader = easyocr.Reader(**reader_params, char_whitelist=num_allowlist)
                    print("char_whitelist 매개변수 적용됨")
                except TypeError:
                    # 어떤 매개변수도 지원되지 않는 경우
                    print("문자 필터 매개변수가 지원되지 않습니다. 필터 없이 진행합니다.")
                    reader = easyocr.Reader(**reader_params)
        else:
            reader = easyocr.Reader(**reader_params)
    
    except Exception as e:
        print(f"EasyOCR 초기화 중 오류 발생: {e}")
        print("기본 설정으로 계속합니다.")
        reader = easyocr.Reader(
            lang_list=langs,
            gpu=torch.cuda.is_available(),
            verbose=False
        )
    
    print("EasyOCR 초기화 완료")
    
    # 이미지 처리
    process_image(args.image, craft_net, reader, 
                 text_threshold=args.text_threshold,
                 link_threshold=args.link_threshold,
                 low_text=args.low_text,
                 output_dir=args.output_dir,
                 allowlist=num_allowlist)

if __name__ == '__main__':
    main()
