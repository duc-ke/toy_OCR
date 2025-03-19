#!/bin/bash

# requirements.sh - CRAFT-OCR 프로젝트 환경 설정 스크립트

echo "=== CRAFT-OCR 환경 설정 시작 ==="

# 필요한 패키지 설치
echo "1. 필요한 패키지 설치 중..."
pip install gdown easyocr torch torchvision numpy opencv-python matplotlib scikit-image

# CRAFT 저장소 클론
echo "2. CRAFT-pytorch 저장소 클론 중..."
if [ ! -d "CRAFT-pytorch" ]; then
    git clone https://github.com/clovaai/CRAFT-pytorch.git
    echo "  - CRAFT-pytorch 저장소 클론 완료"
else
    echo "  - CRAFT-pytorch 저장소가 이미 존재합니다"
fi

# weights 폴더 생성
echo "3. weights 폴더 생성 중..."
mkdir -p CRAFT-pytorch/weights

# 모델 파일 다운로드
echo "4. CRAFT 모델 파일 다운로드 중..."
if [ ! -f "CRAFT-pytorch/weights/craft_mlt_25k.pth" ]; then
    gdown 1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ -O CRAFT-pytorch/weights/craft_mlt_25k.pth
    if [ $? -eq 0 ]; then
        echo "  - 모델 파일 다운로드 완료"
    else
        echo "  - 모델 파일 다운로드 실패"
        echo "  - Google Drive 링크를 확인하고 수동으로 다운로드해주세요"
        echo "  - 링크: https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
    fi
else
    echo "  - 모델 파일이 이미 존재합니다"
fi

# vgg16_bn.py 파일 수정 (모델 URL 문제 해결)
echo "5. VGG16 모델 파일 수정 중..."
if [ -f "CRAFT-pytorch/basenet/vgg16_bn.py" ]; then
    # 백업 파일 생성
    if [ ! -f "CRAFT-pytorch/basenet/vgg16_bn.py.backup" ]; then
        cp CRAFT-pytorch/basenet/vgg16_bn.py CRAFT-pytorch/basenet/vgg16_bn.py.backup
        echo "  - 원본 파일 백업 생성 완료"
    fi
    
    # 파일 수정 (model_urls 업데이트)
    sed -i 's/from torchvision.models.vgg import model_urls/# torchvision 최신 버전 호환을 위한 model_urls 정의\nmodel_urls = {\n    "vgg16_bn": "https:\/\/download.pytorch.org\/models\/vgg16_bn-6c64b313.pth",\n}/' CRAFT-pytorch/basenet/vgg16_bn.py
    echo "  - VGG16 모델 파일 수정 완료"
else
    echo "  - VGG16 모델 파일이 존재하지 않습니다"
fi

echo "=== 설정 완료 ==="
echo "이제 'python ocrtest.py -i [이미지파일]' 명령으로 OCR을 실행할 수 있습니다"