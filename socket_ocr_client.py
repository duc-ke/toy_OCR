#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
소켓 기반 OCR 클라이언트
- 이미지를 전송하고 OCR 결과를 받음
"""

import os
import socket
import struct
import yaml
import time
import cv2
import numpy as np
from enum import Enum
import argparse

# 명령 정의
class Command(Enum):
    Ping = 0
    Ping_Ack = 1
    GetState = 2
    GetState_Ack = 3
    GetProtocolVersion = 4
    GetProtocolVersion_Ack = 5
    SendErrorMessage = 6
    StopServer = 7
    
    # OCR 관련 명령
    LoadOCRModel = 8
    LoadOCRModel_Ack_Pass = 9
    LoadOCRModel_Ack_Fail = 10
    OCRInference = 11
    OCRInference_Ack_Pass = 12
    OCRInference_Ack_Fail = 13
    SendOCRResult = 14

class OCRClient:
    def __init__(self, host='127.0.0.1', port=8870):
        """OCR 클라이언트 초기화"""
        self.server_address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_address)
        self.delimeter = b'\xFF\x00\x00\xFF'
        
        # 명령어 맵핑
        self.command_map = {
            Command.Ping                  : b'\x00\x00',
            Command.Ping_Ack              : b'\x00\x01',
            Command.GetState              : b'\x02\x00',
            Command.GetState_Ack          : b'\x02\x01',
            Command.StopServer            : b'\x05\x00',
            
            Command.GetProtocolVersion    : b'\xFF\x00',
            Command.GetProtocolVersion_Ack: b'\xFF\x01',
            Command.SendErrorMessage      : b'\xEE\x00',
            
            Command.LoadOCRModel          : b'\x10\x00',
            Command.LoadOCRModel_Ack_Pass : b'\x10\x01',
            Command.LoadOCRModel_Ack_Fail : b'\x10\x02',
            Command.OCRInference          : b'\x11\x00',
            Command.OCRInference_Ack_Pass : b'\x11\x01',
            Command.OCRInference_Ack_Fail : b'\x11\x02',
            Command.SendOCRResult         : b'\x12\x00',
        }
        
        print(f"연결됨: {host}:{port}")
    
    def build_message(self, command, data=b''):
        """통신 메시지 구성"""
        output = bytearray()
        output.extend(self.delimeter)
        output.extend(self.command_map[command])
        data_length = len(data)
        output.extend(struct.pack('<I', data_length))
        output.extend(data)
        return bytes(output)
    
    def send_message(self, message):
        """메시지 전송"""
        self.sock.sendall(message)
    
    def receive_response(self):
        """응답 수신"""
        header_length = 10
        header = self._recv_n_bytes(header_length)
        if not header.startswith(self.delimeter):
            raise ValueError("응답에 유효한 구분자가 없습니다")
        
        command_bytes = header[4:6]
        data_length = struct.unpack('<I', header[6:10])[0]
        
        data = b''
        if data_length > 0:
            data = self._recv_n_bytes(data_length)
        
        command = self._parse_command(command_bytes)
        return command, data
    
    def _recv_n_bytes(self, n):
        """지정된 바이트 수만큼 수신"""
        data = b''
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("데이터 수신 중 연결이 끊겼습니다")
            data += packet
        return data
    
    def _parse_command(self, command_bytes):
        """명령어 바이트를 Enum으로 변환"""
        for cmd, cmd_bytes in self.command_map.items():
            if cmd_bytes == command_bytes:
                return cmd
        return None
    
    def ping(self):
        """핑 테스트"""
        message = self.build_message(Command.Ping)
        self.send_message(message)
        response_cmd, response_data = self.receive_response()
        if response_cmd == Command.Ping_Ack:
            print("핑 성공")
            return True
        else:
            print("핑 실패")
            return False
    
    def get_protocol_version(self):
        """프로토콜 버전 확인"""
        message = self.build_message(Command.GetProtocolVersion)
        self.send_message(message)
        response_cmd, response_data = self.receive_response()
        if response_cmd == Command.GetProtocolVersion_Ack:
            version = response_data.decode('utf-8')
            print(f"프로토콜 버전: {version}")
            return version
        else:
            print("프로토콜 버전 확인 실패")
            return None
    
    def get_state(self):
        """서버 상태 확인"""
        message = self.build_message(Command.GetState)
        self.send_message(message)
        response_cmd, response_data = self.receive_response()
        if response_cmd == Command.GetState_Ack:
            state = response_data.decode('utf-8')
            print(f"서버 상태: {state}")
            return state
        else:
            print(f"서버 상태 확인 실패: {response_cmd}")
            return None
    
    def load_ocr_model(self, config=None):
        """OCR 모델 로드"""
        if config is None:
            config = {
                'craft_model_path': 'CRAFT-pytorch/weights/craft_mlt_25k.pth',
                'lang_list': ['ko', 'en']
            }
        
        config_yaml_str = yaml.dump(config)
        data = config_yaml_str.encode('utf-8')
        message = self.build_message(Command.LoadOCRModel, data)
        self.send_message(message)
        
        response_cmd, response_data = self.receive_response()
        if response_cmd == Command.LoadOCRModel_Ack_Pass:
            print("OCR 모델 로드 성공")
            return True
        else:
            print("OCR 모델 로드 실패")
            return False
    
    def ocr_image(self, image_path=None, image=None, wait_for_result=True, image_format='.png'):
        """이미지 OCR 처리
        
        Args:
            image_path: 처리할 이미지 파일 경로
            image: 이미지 객체 (NumPy 배열)
            wait_for_result: 결과를 기다릴지 여부
            image_format: 이미지 인코딩 형식 ('.png', '.jpg' 등)
        
        Returns:
            OCR 결과 텍스트 또는 처리 상태
        """
        try:
            # 이미지 준비
            if image is None and image_path is not None:
                print(f"이미지 로드 중: {image_path}")
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
            
            if image is None:
                raise ValueError("이미지가 제공되지 않았습니다")
            
            # 이미지 크기 및 형식 확인
            h, w = image.shape[:2]
            print(f"이미지 크기: {w}x{h} 픽셀, 형식: {image.dtype}")
            
            # 너무 큰 이미지 리사이징 (최대 1920x1080)
            max_width = 1920
            max_height = 1080
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                print(f"이미지 리사이징: {w}x{h} -> {new_w}x{new_h}")
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 이미지 인코딩 (PNG는 무손실이지만 큼, JPEG는 손실 압축으로 작음)
            if image_format.lower() == '.png':
                _, img_encoded = cv2.imencode('.png', image)
            else:
                # JPEG 압축 품질 설정 (0-100)
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                _, img_encoded = cv2.imencode('.jpg', image, encode_params)
            
            img_bytes = img_encoded.tobytes()
            img_size_kb = len(img_bytes) / 1024
            print(f"이미지 인코딩 완료: {img_size_kb:.1f} KB")
            
            # 이미지 전송
            try:
                message = self.build_message(Command.OCRInference, img_bytes)
                self.send_message(message)
                print(f"이미지 전송 완료: {img_size_kb:.1f} KB")
            except Exception as e:
                raise ConnectionError(f"이미지 전송 실패: {e}")
            
            # 응답 확인
            try:
                response_cmd, response_data = self.receive_response()
            except Exception as e:
                raise ConnectionError(f"서버 응답 수신 실패: {e}")
            
            if response_cmd != Command.OCRInference_Ack_Pass:
                error_msg = response_data.decode('utf-8') if response_data else "알 수 없는 오류"
                raise ValueError(f"OCR 처리 요청 실패: {error_msg}")
            
            print("OCR 처리 요청 수락됨, 처리 중...")
            
            # 결과 대기
            if wait_for_result:
                try:
                    response_cmd, response_data = self.receive_response()
                    if response_cmd == Command.SendOCRResult:
                        ocr_result = response_data.decode('utf-8')
                        print("OCR 결과 수신 완료")
                        return ocr_result
                    else:
                        error_msg = response_data.decode('utf-8') if response_data else f"예상치 못한 응답: {response_cmd}"
                        raise ValueError(f"OCR 결과 수신 실패: {error_msg}")
                except Exception as e:
                    raise ConnectionError(f"결과 수신 중 오류: {e}")
            
            return True
        
        except Exception as e:
            print(f"OCR 처리 오류: {e}")
            return None
    
    def stop_server(self):
        """서버 종료 요청"""
        message = self.build_message(Command.StopServer)
        self.send_message(message)
        print("서버 종료 요청 전송됨")
    
    def close(self):
        """클라이언트 종료"""
        self.sock.close()
        print("클라이언트 연결 종료")

def parse_ocr_csv_result(csv_data):
    """
    CSV 형식의 OCR 결과를 파싱하여 각 텍스트 영역의 정보를 추출
    
    Args:
        csv_data: OCR 서버에서 반환된 CSV 형식 문자열
        
    Returns:
        튜플: (텍스트 영역 정보 목록, 전체 텍스트)
    """
    # 텍스트 영역 정보를 저장할 리스트
    regions = []
    full_text = ""
    
    try:
        # CSV 형식 파싱
        lines = csv_data.strip().split('\n')
        
        # 헤더와 데이터 분리
        header = None
        data_lines = []
        
        for line in lines:
            if line.startswith('region_id,text,confidence'):
                header = line
                continue
            elif not line.strip() or line.startswith('---'):
                continue
            
            # 데이터 라인 추가
            if ',' in line:
                parts = line.strip().split(',')
                
                # 첫 번째 필드가 region_id
                try:
                    region_id = int(parts[0])
                    
                    # region_id가 -999이면 전체 텍스트를 의미
                    if region_id == -999:
                        full_text = parts[1]
                        continue
                    
                    # 일반적인 텍스트 영역 정보
                    data_lines.append(line)
                except ValueError:
                    # region_id가 정수가 아닌 경우 무시
                    continue
        
        # 각 영역 정보 파싱
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 11:  # 최소 11개 필드 (region_id, text, confidence, x1,y1,x2,y2,x3,y3,x4,y4)
                try:
                    region_id = int(parts[0])
                    text = parts[1]
                    confidence = float(parts[2])
                    
                    # 좌표 파싱 (x1,y1,x2,y2,x3,y3,x4,y4)
                    coords = []
                    for i in range(3, 11):
                        coords.append(int(float(parts[i])))
                    
                    # 점 형태로 변환 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    points = []
                    for i in range(0, 8, 2):
                        points.append((coords[i], coords[i+1]))
                    
                    # 영역 정보 추가
                    regions.append({
                        'id': region_id,
                        'text': text,
                        'confidence': confidence,
                        'points': points
                    })
                except (ValueError, IndexError) as e:
                    print(f"CSV 라인 파싱 오류: {line} - {e}")
    
    except Exception as e:
        print(f"OCR 결과 파싱 오류: {e}")
    
    return regions, full_text

def visualize_ocr_results(image_path, regions, output_path=None):
    """
    OCR 결과를 이미지에 시각화
    
    Args:
        image_path: 원본 이미지 경로
        regions: 파싱된 텍스트 영역 정보 리스트
        output_path: 결과 이미지 저장 경로
    
    Returns:
        시각화된 이미지
    """
    try:
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return None
        
        # 결과 이미지 생성
        result_img = image.copy()
        
        # 색상 설정
        box_color = (0, 255, 0)  # 초록색 (BGR)
        text_bg_color = (0, 0, 0)  # 검은색
        text_color = (255, 255, 255)  # 흰색
        
        # 각 텍스트 영역에 대해
        for region in regions:
            # 폴리곤 좌표를 정수형 배열로 변환
            points = np.array(region['points'], dtype=np.int32)
            
            # 다각형 그리기
            cv2.polylines(result_img, [points], True, box_color, 2)
            
            # 텍스트 영역 ID와 신뢰도 표시
            region_id = region['id']
            confidence = region['confidence']
            
            # 다각형의 중심 좌표 계산
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            
            # ID 배경 원형 그리기
            cv2.circle(result_img, (center_x, center_y), 15, box_color, -1)
            
            # ID 텍스트
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            cv2.putText(result_img, str(region_id), (center_x-5, center_y+5), 
                        font, font_scale, text_color, thickness)
            
            # 텍스트 및 신뢰도 표시
            text = region['text']
            
            # 텍스트 위치 계산 - 다각형 상단 위치
            top_y = min([p[1] for p in points]) - 10
            left_x = min([p[0] for p in points])
            
            # 텍스트 크기 측정
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 배경 사각형 그리기
            cv2.rectangle(result_img, 
                         (left_x, top_y - text_height - 5), 
                         (left_x + text_width, top_y + 5), 
                         text_bg_color, -1)
            
            # 텍스트 그리기
            cv2.putText(result_img, text, (left_x, top_y), 
                       font, font_scale, text_color, thickness)
        
        # 결과 이미지 저장
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, result_img)
            print(f"결과 이미지 저장: {output_path}")
        
        return result_img
    
    except Exception as e:
        print(f"결과 시각화 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def ocr_test(image_path='jupyters/testimgocr.png', check_model=False, host='127.0.0.1', port=8870, image_format='.jpg', save_visualization=True):
    """OCR 테스트"""
    client = OCRClient(host=host, port=port)
    
    try:
        # 서버 상태 확인
        client.ping()
        client.get_protocol_version()
        state = client.get_state()
        
        # OCR 모델 로드 확인 (선택적)
        model_loaded = False
        if state and 'ModelLoaded:True' in state:
            model_loaded = True
            print("서버에 OCR 모델이 이미 로드되어 있습니다.")
        
        # 모델이 로드되지 않았거나 check_model이 True인 경우에만 로드 요청
        if check_model and not model_loaded:
            print("OCR 모델 로드 요청 중...")
            config = {
                'craft_model_path': 'CRAFT-pytorch/weights/craft_mlt_25k.pth',
                'lang_list': ['en']  # 언어 목록 - 서버와 동일하게 설정
            }
            model_loaded = client.load_ocr_model(config)
        
        # 이미지 OCR 처리
        if model_loaded or not check_model:  # 모델 체크를 하지 않거나, 모델이 로드되었으면 진행
            # 이미지 존재 확인
            if not os.path.exists(image_path):
                print(f"경고: 이미지 파일이 존재하지 않습니다: {image_path}")
                return None
                
            print(f"이미지 처리 중: {image_path}")
            result = client.ocr_image(image_path=image_path, image_format=image_format)
            
            if result:
                print("\n=== OCR 결과 ===")
                
                # 출력 디렉토리 생성
                output_dir = 'jupyters/output'
                os.makedirs(output_dir, exist_ok=True)
                
                # 결과 텍스트 저장
                with open(os.path.join(output_dir, 'ocr_result.csv'), 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"결과가 '{os.path.join(output_dir, 'ocr_result.csv')}'에 저장되었습니다.")
                
                # CSV 데이터 파싱
                regions, full_text = parse_ocr_csv_result(result)
                print(f"인식된 텍스트 영역: {len(regions)}")
                print(f"전체 텍스트: {full_text}")
                
                # 결과 시각화 및 저장
                if save_visualization and regions:
                    output_path = os.path.join(output_dir, 'ocr_visualization.jpg')
                    visualize_ocr_results(image_path, regions, output_path)
                
                return result
            else:
                print("OCR 처리 실패")
        else:
            print("모델이 로드되지 않아 처리를 진행할 수 없습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        client.close()
    
    return None

def batch_ocr_test(image_dir, check_model=False, host='127.0.0.1', port=8870, image_format='.jpg', save_visualization=True):
    """디렉토리 내 모든 이미지에 대한 OCR 테스트"""
    client = OCRClient(host=host, port=port)
    
    try:
        # 서버 상태 확인
        client.ping()
        state = client.get_state()
        
        # OCR 모델 로드 확인 (선택적)
        model_loaded = False
        if state and 'ModelLoaded:True' in state:
            model_loaded = True
            print("서버에 OCR 모델이 이미 로드되어 있습니다.")
        
        # 모델이 로드되지 않았거나 check_model이 True인 경우에만 로드 요청
        if check_model and not model_loaded:
            print("OCR 모델 로드 요청 중...")
            config = {
                'craft_model_path': 'CRAFT-pytorch/weights/craft_mlt_25k.pth',
                'lang_list': ['en']  # 언어 목록 - 서버와 동일하게 설정
            }
            model_loaded = client.load_ocr_model(config)
        
        # 이미지 OCR 처리
        if model_loaded or not check_model:  # 모델 체크를 하지 않거나, 모델이 로드되었으면 진행
            # 이미지 목록 가져오기
            image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                print(f"{image_dir}에 이미지 파일이 없습니다.")
                return
            
            # 결과 디렉토리 생성
            result_dir = os.path.join(image_dir, 'ocr_results')
            os.makedirs(result_dir, exist_ok=True)
            
            # 각 이미지 처리
            for image_file in image_files:
                image_path = os.path.join(image_dir, image_file)
                print(f"\n{image_file} 처리 중...")
                
                result = client.ocr_image(image_path=image_path, image_format=image_format)
                if result:
                    # 파일명에서 확장자 제외
                    base_name = os.path.splitext(image_file)[0]
                    
                    # 결과 CSV 저장
                    result_csv_file = os.path.join(result_dir, f"{base_name}_ocr.csv")
                    with open(result_csv_file, 'w', encoding='utf-8') as f:
                        f.write(result)
                    print(f"결과가 {result_csv_file}에 저장되었습니다.")
                    
                    # CSV 파싱
                    regions, full_text = parse_ocr_csv_result(result)
                    print(f"인식된 텍스트 영역: {len(regions)}")
                    
                    # 결과 시각화 및 저장
                    if save_visualization and regions:
                        vis_output_path = os.path.join(result_dir, f"{base_name}_ocr_vis.jpg")
                        visualize_ocr_results(image_path, regions, vis_output_path)
            
            print("\n모든 이미지 처리 완료")
        else:
            print("모델이 로드되지 않아 처리를 진행할 수 없습니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    # 인자 처리
    parser = argparse.ArgumentParser(description='OCR 클라이언트')
    parser.add_argument('--image', '-i', type=str, help='처리할 이미지 경로')
    parser.add_argument('--dir', '-d', type=str, help='처리할 이미지 디렉토리')
    parser.add_argument('--check-model', '-c', action='store_true', help='모델 로드 상태 확인 및 필요 시 로드 요청')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='OCR 서버 호스트 주소')
    parser.add_argument('--port', type=int, default=8870, help='OCR 서버 포트 번호')
    parser.add_argument('--format', type=str, choices=['.png', '.jpg'], default='.jpg', 
                        help='이미지 전송 형식 (png=무손실/크기큼, jpg=손실압축/크기작음)')
    parser.add_argument('--no-vis', action='store_true', help='시각화 결과 저장 비활성화')
    
    args = parser.parse_args()
    
    # 시각화 여부
    save_visualization = not args.no_vis
    
    # 기본 이미지 경로
    default_image_path = 'jupyters/testimgocr.png'
    
    # 기본 이미지가 없으면 대체 경로 시도
    if not os.path.exists(default_image_path):
        alt_paths = [
            'testimgocr.png',  # 현재 디렉토리
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testimgocr.png'),  # 스크립트 위치
            'jupyters/1-1.testimgocr2.JPG',  # 대체 파일
            'jupyters/test_images/testimgocr.png'  # 테스트 디렉토리
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                default_image_path = path
                print(f"기본 이미지를 찾았습니다: {default_image_path}")
                break
        else:
            print("경고: 기본 테스트 이미지를 찾을 수 없습니다. 이미지 경로를 수동으로 지정하세요.")
    
    if args.dir:
        # 디렉토리 내 모든 이미지 OCR 테스트
        batch_ocr_test(args.dir, check_model=args.check_model, 
                      host=args.host, port=args.port, image_format=args.format,
                      save_visualization=save_visualization)
    elif args.image:
        # 단일 이미지 OCR 테스트
        ocr_test(args.image, check_model=args.check_model,
                host=args.host, port=args.port, image_format=args.format,
                save_visualization=save_visualization)
    else:
        # 기본 테스트 이미지 사용
        ocr_test(default_image_path, check_model=args.check_model,
                host=args.host, port=args.port, image_format=args.format,
                save_visualization=save_visualization) 