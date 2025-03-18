#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
소켓 기반 OCR 서버
- ocrtest.py의 OCR 기능을 소켓 통신으로 제공
- 이미지를 받아 OCR 처리 후 결과를 반환
"""

import os
import sys
import io
import time
import yaml
import numpy as np
import asyncio
import struct
import gc
import socket
import threading
from threading import Thread
import cv2
import torch
from enum import Enum, auto
from datetime import datetime

# ocrtest.py 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ocrtest

# 로깅 설정
def print_log(message, type="info"):
    print(f"[{type.upper()}] {message}", flush=True)

# 명령 정의
class Command(Enum):
    Ping = auto()
    Ping_Ack = auto()
    GetState = auto()
    GetState_Ack = auto()
    GetProtocolVersion = auto()
    GetProtocolVersion_Ack = auto()
    SendErrorMessage = auto()
    StopServer = auto()
    
    # OCR 관련 명령
    LoadOCRModel = auto()
    LoadOCRModel_Ack_Pass = auto()
    LoadOCRModel_Ack_Fail = auto()
    OCRInference = auto()
    OCRInference_Ack_Pass = auto()
    OCRInference_Ack_Fail = auto()
    SendOCRResult = auto()

# 서버 상태 정의
class State(Enum):
    Ready = auto()
    LoadingOCRModel = auto()
    OCRProcessing = auto()

# 서버 종료 예외
class ServerStopException(Exception):
    pass

# 프로토콜 구현
class OCRProtocol:
    """OCR 서비스 프로토콜 정의 클래스"""
    
    def __init__(self):
        self.delimeter = b'\xFF\x00\x00\xFF'
        self.version = "OCR_1.0"
        self.session = None
        self.state = State.Ready
        self.state_lock = threading.Lock()
        
        # OCR 관련 변수
        self.craft_net = None
        self.reader = None
        self.model_loaded = False
        
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
    
    def _handle_command(self, command, data):
        """명령어 처리"""
        if command == Command.Ping:
            self._send_message(Command.Ping_Ack)
            
        elif command == Command.GetProtocolVersion:
            self._send_message(Command.GetProtocolVersion_Ack, self.version)
            
        elif command == Command.GetState:
            with self.state_lock:
                state_str = f"ServerState:{self.state.name},ModelLoaded:{self.model_loaded}"
            self._send_message(Command.GetState_Ack, state_str)
            
        elif command == Command.StopServer:
            self.session.server.stop_server()
            raise ServerStopException("Server is stopping...")
            
        elif command == Command.LoadOCRModel:
            # 이미 모델이 서버에 로드되어 있으므로 성공 응답 반환
            print_log("모델 로드 요청 수신 - 모델이 이미 로드되어 있습니다", "info")
            self._send_message(Command.LoadOCRModel_Ack_Pass)
            
        elif command == Command.OCRInference:
            self._handle_ocr_inference(data)
    
    def _handle_ocr_inference(self, data):
        """OCR 추론 처리"""
        try:
            # 모델 로드 여부 확인
            if not self.model_loaded:
                print_log("OCR 모델이 로드되지 않았습니다", "error")
                self._send_message(Command.OCRInference_Ack_Fail)
                return
            
            # 서버 상태 확인
            with self.state_lock:
                if self.state != State.Ready:
                    print_log("서버가 사용 중입니다. 지금은 OCR을 수행할 수 없습니다.", "error")
                    self._send_message(Command.OCRInference_Ack_Fail)
                    return
                self.state = State.OCRProcessing
            
            # 이미지 데이터 유효성 검사
            if not data or len(data) < 16:  # 최소 헤더 크기 확인
                print_log(f"유효하지 않은 이미지 데이터: 크기 {len(data) if data else 0} 바이트", "error")
                self._send_message(Command.OCRInference_Ack_Fail)
                with self.state_lock:
                    self.state = State.Ready
                return
            
            # 확인 응답 전송
            self._send_message(Command.OCRInference_Ack_Pass)
            
            # 이미지 디코딩
            try:
                # 헤더에서 B, C, H, W 값 추출
                B, C, H, W = struct.unpack('<IIII', data[:16])
                print_log(f"이미지 형태: ({B}, {C}, {H}, {W})", "info")
                
                # 나머지 데이터를 NumPy 배열로 변환
                img_data = np.frombuffer(data[16:], dtype=np.uint8).reshape(B, C, H, W)
                
                # 첫 번째 이미지만 사용 (배치가 1보다 큰 경우)
                img_data = img_data[0]  # (C, H, W)
                
                # 채널 순서 변경 (C, H, W) -> (H, W, C)
                img = np.transpose(img_data, (1, 2, 0))
                
                # 이미지 유효성 검사
                if img is None or img.size == 0:
                    print_log("이미지 데이터가 유효하지 않습니다", "error")
                    with self.state_lock:
                        self.state = State.Ready
                    return
                
                # 이미지 크기 및 형식 출력
                h, w = img.shape[:2]
                print_log(f"이미지 변환 완료: {w}x{h} 픽셀, {img.dtype}", "info")
                
                # OCR 처리를 별도 쓰레드에서 실행
                Thread(target=self._process_ocr, args=(img,)).start()
                
            except Exception as e:
                print_log(f"이미지 디코딩 오류: {e}", "error")
                import traceback
                print_log(traceback.format_exc(), "error")
                with self.state_lock:
                    self.state = State.Ready
                self._send_message(Command.OCRInference_Ack_Fail)
            
        except Exception as e:
            print_log(f"OCR 추론 중 오류 발생: {e}", "error")
            with self.state_lock:
                self.state = State.Ready
            self._send_message(Command.OCRInference_Ack_Fail)
    
    def _process_ocr(self, img):
        """OCR 처리 실행 (쓰레드에서 실행)"""
        try:
            # 임시 디렉토리 생성
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_ocr_output")
            try:
                os.makedirs(output_dir, exist_ok=True)
                print_log(f"임시 출력 디렉토리 생성/확인: {output_dir}", "info")
            except Exception as e:
                print_log(f"임시 디렉토리 생성 실패: {e}", "error")
                output_dir = 'temp_ocr_output'  # 기본 폴더로 대체
                os.makedirs(output_dir, exist_ok=True)
            
            # OCR 실행
            print_log("이미지 OCR 처리 시작...", "info")
            start_time = time.time()
            
            try:
                img_result, recognized_texts, corrected_images, boxes, polys, confidences = ocrtest.process_image(
                    img,  # 이미지 직접 전달
                    self.craft_net, 
                    self.reader,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4,
                    output_dir=output_dir,
                    direct_img=True,  # 이미지 경로가 아닌 이미지 객체 직접 전달
                    return_coords=True,  # 좌표 정보도 반환
                    return_confidence=True  # 신뢰도 정보도 반환
                )
                
                # 처리 시간 측정
                process_time = time.time() - start_time
                print_log(f"OCR 처리 완료, 소요 시간: {process_time:.2f}초", "info")
                
                # 인식된 텍스트 수 출력
                text_count = sum(1 for t in recognized_texts if t)
                print_log(f"인식된 텍스트 영역: {text_count}/{len(recognized_texts)}", "info")
                
                # 결과 텍스트 생성 - CSV 형식으로 변환
                results_text = self._format_ocr_results(recognized_texts, boxes, polys, confidences)
                
                # 결과 전송
                self._send_message(Command.SendOCRResult, results_text)
                print_log("OCR 결과 전송 완료", "info")
                
            except Exception as e:
                print_log(f"OCR 처리 중 오류 발생: {str(e)}", "error")
                import traceback
                print_log(traceback.format_exc(), "error")
                # 간단한 오류 메시지 클라이언트에 전송
                error_text = f"OCR 처리 오류: {str(e)}"
                self._send_message(Command.SendOCRResult, error_text)
                
        except Exception as e:
            print_log(f"OCR 처리 쓰레드 오류: {str(e)}", "error")
        finally:
            # 상태 복원
            with self.state_lock:
                self.state = State.Ready
            print_log("OCR 처리 상태 초기화 완료", "info")
    
    def _format_ocr_results(self, recognized_texts, boxes, polys, confidences):
        """OCR 결과 CSV 포맷으로 변환"""
        # CSV 헤더 추가
        result = "region_id,text,confidence,x1,y1,x2,y2,x3,y3,x4,y4\n"
        
        # 각 영역의 결과 추가
        for i, (text, box, poly, confidence) in enumerate(zip(recognized_texts, boxes, polys, confidences)):
            # 텍스트가 있는 경우만 처리
            if text:
                # 인식된 텍스트 정보
                region_id = i + 1
                
                # 폴리곤 좌표 처리 - 네 모서리 좌표
                try:
                    coords = []
                    
                    # 다각형이 None이 아니고 유효한 좌표가 있는 경우
                    if poly is not None and len(poly) >= 4:
                        # poly는 [x,y] 좌표의 배열
                        try:
                            coords = poly.reshape(-1).tolist()  # 1차원 배열로 변환
                        except:
                            # 리셰이프 오류 시 직접 변환 시도
                            coords = []
                            for point in poly:
                                if isinstance(point, np.ndarray):
                                    coords.extend(point.tolist())
                                else:
                                    coords.extend([point[0], point[1]])
                        
                        # 좌표가 4개보다 많은 경우 (다각형), 첫 4개 점만 사용
                        if len(coords) > 8:  # x,y 좌표 쌍이 4개 이상
                            # 사각형 형태로 근사화
                            poly_np = np.array(poly, dtype=np.float32)
                            rect = cv2.minAreaRect(poly_np)
                            box_points = cv2.boxPoints(rect)
                            box_points = np.int0(box_points)
                            coords = box_points.reshape(-1).tolist()
                    
                    # 좌표가 없거나 부족한 경우 박스 사용
                    if not coords or len(coords) < 8:
                        # 박스 좌표 사용
                        try:
                            if box is not None:
                                box_coords = box.reshape(-1).tolist()
                                coords = box_coords
                        except:
                            # 리셰이프 오류 시 직접 변환 시도
                            coords = []
                            for point in box:
                                if isinstance(point, np.ndarray):
                                    coords.extend(point.tolist())
                                else:
                                    coords.extend([point[0], point[1]])
                    
                    # 좌표가 여전히 부족하면 더미 좌표 사용
                    if not coords or len(coords) < 8:
                        # 더미 좌표 (0,0,0,0,0,0,0,0)
                        print_log(f"영역 {region_id}의 좌표가 불완전합니다. 더미 좌표를 사용합니다.", "warn")
                        coords = [0, 0, 0, 0, 0, 0, 0, 0]
                    
                    # 정확히 8개의 좌표(x1,y1,x2,y2,x3,y3,x4,y4)가 되도록 조정
                    if len(coords) > 8:
                        coords = coords[:8]  # 처음 8개 좌표만 사용
                    
                    # CSV 행 추가
                    coord_str = ",".join(map(str, coords))
                    result += f"{region_id},{text},{confidence:.2f},{coord_str}\n"
                
                except Exception as e:
                    print_log(f"좌표 처리 중 오류: {str(e)}", "error")
                    # 오류 발생 시 빈 좌표로 대체
                    result += f"{region_id},{text},{confidence:.2f},0,0,0,0,0,0,0,0\n"
        
        # 전체 텍스트를 CSV 형식으로 추가 (region_id를 -999로 설정)
        full_text = " ".join([t for t in recognized_texts if t])
        if full_text:
            result += f"-999,{full_text},1.00,0,0,0,0,0,0,0,0\n"
        
        return result
    
    def receive_data(self, session):
        """데이터 수신 시작"""
        self.session = session
        Thread(target=self._receive_data).start()
    
    def _receive_data(self):
        """데이터 수신 처리 (쓰레드에서 실행)"""
        stream = self.session.client
        buffer = b''
        delimeter_length = len(self.delimeter)
        
        while True:
            try:
                # 한 바이트씩 읽어 delimeter 확인
                byte = self._read_exactly(stream, 1)
                buffer += byte
                
                # 버퍼가 delimeter 이상이면 검사
                if len(buffer) >= delimeter_length:
                    # delimeter와 일치하는지 확인
                    if buffer[-delimeter_length:] == self.delimeter:
                        # 헤더의 나머지 부분 읽기
                        header_remaining = self._read_exactly(stream, 10 - delimeter_length)
                        command_bytes = header_remaining[0:2]
                        data_length = struct.unpack('<I', header_remaining[2:6])[0]
                        
                        # 데이터 읽기
                        data = self._read_exactly(stream, data_length) if data_length > 0 else b''
                        
                        # 명령어 파싱 및 처리
                        command = self._parse_command(command_bytes)
                        if command:
                            self._handle_command(command, data)
                        else:
                            print_log(f"Unknown command: {command_bytes.hex()}", "error")
                        
                        # 버퍼 초기화
                        buffer = b''
                    else:
                        # delimeter가 중간에 일치하지 않는 경우, 버퍼 유지
                        buffer = buffer[-delimeter_length + 1:]
                
            except ServerStopException as e:
                print_log(f"ServerStopException caught: {e}", "info")
                break
                
            except ConnectionError:
                print_log("Connection closed", "info")
                self.session.server.client_disconnected(self.session)
                break
                
            except Exception as e:
                print_log(f"Error in receive_data: {e}", "error")
                self.session.server.client_disconnected(self.session)
                break
    
    def _read_exactly(self, stream, length):
        """지정된 길이의 데이터를 정확히 읽음"""
        data = b''
        while len(data) < length:
            packet = stream.recv(length - len(data))
            if not packet:
                raise ConnectionError("Connection lost while reading data")
            data += packet
        return data
    
    def _parse_command(self, command_bytes):
        """바이트 명령어를 Enum으로 변환"""
        for cmd, cmd_bytes in self.command_map.items():
            if cmd_bytes == command_bytes:
                return cmd
        return None
    
    def _send_message(self, command, data_str=None):
        """메시지 전송"""
        try:
            stream = self.session.client
            message = self._construct_message(command, data_str)
            stream.sendall(message)
        except Exception as e:
            print_log(f"Error sending message: {e}", "error")
    
    def _construct_message(self, command, data_str=None):
        """메시지 구성"""
        output = bytearray()
        output.extend(self.delimeter)
        output.extend(self.command_map[command])
        
        if data_str is not None:
            # 문자열인 경우 UTF-8로 인코딩
            if isinstance(data_str, str):
                data_bytes = data_str.encode('utf-8')
            else:
                data_bytes = data_str
            
            # 데이터 길이 추가
            output.extend(struct.pack('<I', len(data_bytes)))
            output.extend(data_bytes)
        else:
            # 데이터 없음
            output.extend(struct.pack('<I', 0))
        
        return bytes(output)

# 클라이언트 세션
class ClientSession:
    def __init__(self, client, server):
        self.client = client
        self.server = server
        self.protocol = OCRProtocol()
        
        # 서버에서 로드된 모델 전달
        self.protocol.craft_net = server.craft_net
        self.protocol.reader = server.reader
        self.protocol.model_loaded = server.model_loaded
    
    def is_connected(self):
        try:
            # 연결 상태 확인
            self.client.settimeout(0.1)
            self.client.recv(1, socket.MSG_PEEK)
            self.client.settimeout(None)
            return True
        except socket.timeout:
            self.client.settimeout(None)
            return True
        except:
            return False
    
    def session_start(self):
        self.protocol.receive_data(self)

# OCR 서버
class OCRServer:
    def __init__(self, host="127.0.0.1", port=8870, craft_model_path='CRAFT-pytorch/weights/craft_mlt_25k.pth', lang_list=None):
        self.server_address = (host, port)
        self.clients = []
        self.client_count = 0
        self.server_running = True
        self.server_stop_lock = threading.Lock()
        
        # OCR 모델 관련 변수
        self.craft_model_path = craft_model_path
        self.lang_list = lang_list or ['en'] # ['ko', 'en']
        self.craft_net = None
        self.reader = None
        self.model_loaded = False
        
        # 서버 시작 시 모델 로드
        print_log("서버 초기화 중 - OCR 모델 로드 시작", "info")
        self._load_ocr_models()
    
    def _load_ocr_models(self):
        """OCR 모델 로드"""
        try:
            # CRAFT 모델 로드
            print_log(f"CRAFT 모델 로드 중... {self.craft_model_path}")
            self.craft_net = ocrtest.load_craft_model(self.craft_model_path)
            
            # EasyOCR 초기화
            print_log(f"EasyOCR 초기화 중... (언어: {', '.join(self.lang_list)})")
            
            # 리더 파라미터 설정
            reader_params = {
                'lang_list': self.lang_list,
                'gpu': torch.cuda.is_available(),
                'quantize': False,
                'verbose': False
            }
            
            try:
                import easyocr
                self.reader = easyocr.Reader(**reader_params)
                self.model_loaded = True
                print_log("OCR 모델 로드 완료!", "info")
            except Exception as e:
                print_log(f"EasyOCR 초기화 오류: {e}", "error")
                raise
                
        except Exception as e:
            print_log(f"OCR 모델 로드 실패: {e}", "error")
            self.model_loaded = False
    
    def start_server(self):
        """서버 시작"""
        # 모델이 로드되었는지 확인
        if not self.model_loaded:
            print_log("OCR 모델이 로드되지 않았습니다. 서버를 시작할 수 없습니다.", "error")
            return False
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(5)  # 최대 5개 클라이언트 연결 허용
        self.server_socket.settimeout(5)  # 타임아웃 설정
        
        print_log(f"OCR 서버 시작 - {self.server_address[0]}:{self.server_address[1]}")
        
        def accept_clients():
            print_log("클라이언트 대기 중...")
            while self.server_running:
                try:
                    client, client_address = self.server_socket.accept()
                    self.client_count += 1
                    session = ClientSession(client, self)
                    session.session_start()
                    self.clients.append(session)
                    print_log(f"클라이언트 연결됨: {client_address}")
                except socket.timeout:
                    continue  # 타임아웃 발생 시 루프 계속
                except OSError:
                    break     # 서버 소켓 닫힌 경우
            
        self.accept_thread = Thread(target=accept_clients)
        self.accept_thread.start()
        return True
    
    def client_disconnected(self, session):
        """클라이언트 연결 해제 처리"""
        with self.server_stop_lock:
            try:
                if session in self.clients:
                    self.clients.remove(session)
                    self.client_count -= 1
                    print_log(f"Client disconnected. {self.client_count} clients remaining")
            except Exception as e:
                print_log(f"Error removing client: {e}", "error")
    
    def stop_server(self):
        """서버 종료"""
        with self.server_stop_lock:
            print_log("Stopping server...")
            self.server_running = False
            
            # 클라이언트 연결 종료
            for session in self.clients[:]:  # 복사본 사용
                try:
                    session.client.close()
                except:
                    pass
            
            # 서버 소켓 종료
            try:
                self.server_socket.close()
            except:
                pass
            
            # 쓰레드 종료 대기
            if hasattr(self, 'accept_thread') and self.accept_thread.is_alive():
                self.accept_thread.join()
                
            print_log("Server stopped")

# ocrtest.py 수정: 이미지 객체 직접 사용 지원
def process_image_patch():
    """OCR 처리 함수 패치: 이미지 경로 대신 이미지 객체 직접 받을 수 있도록 패치"""
    
    # 원본 함수 복사
    original_process_image = ocrtest.process_image
    
    # 새 함수 정의
    def patched_process_image(image_path_or_img, craft_net, reader, 
                            text_threshold=0.7, link_threshold=0.4, 
                            low_text=0.4, canvas_size=1280, 
                            mag_ratio=1.5, poly=True, 
                            output_dir='output',
                            paragraph=False, detail=1, allowlist=None,
                            direct_img=False, return_coords=False, return_confidence=False):
        """
        이미지 처리 함수: 텍스트 영역 감지, 기울기 보정, OCR 수행
        
        direct_img=True일 경우 image_path_or_img는 이미지 객체로 간주
        return_coords=True일 경우 좌표 정보도 함께 반환
        return_confidence=True일 경우 신뢰도 정보도 함께 반환
        """
        
        # 이미지 객체인 경우 임시 파일로 저장
        if direct_img:
            print_log("이미지 객체를 직접 처리합니다", "info")
            
            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)
            
            # 임시 파일 경로 생성
            temp_img_path = os.path.join(output_dir, f"temp_img_{int(time.time())}.png")
            
            try:
                # 이미지 저장
                cv2.imwrite(temp_img_path, image_path_or_img)
                print_log(f"이미지를 임시 파일로 저장했습니다: {temp_img_path}", "info")
                
                # 원본 함수 호출 (파일 경로 사용)
                if return_coords and return_confidence:
                    result = original_process_image(
                        temp_img_path, craft_net, reader, 
                        text_threshold, link_threshold, 
                        low_text, canvas_size, 
                        mag_ratio, poly, 
                        output_dir, paragraph, detail, allowlist,
                        direct_img=False, return_coords=True, return_confidence=True
                    )
                elif return_coords:
                    result = original_process_image(
                        temp_img_path, craft_net, reader, 
                        text_threshold, link_threshold, 
                        low_text, canvas_size, 
                        mag_ratio, poly, 
                        output_dir, paragraph, detail, allowlist,
                        direct_img=False, return_coords=True, return_confidence=False
                    )
                elif return_confidence:
                    result = original_process_image(
                        temp_img_path, craft_net, reader, 
                        text_threshold, link_threshold, 
                        low_text, canvas_size, 
                        mag_ratio, poly, 
                        output_dir, paragraph, detail, allowlist,
                        direct_img=False, return_coords=False, return_confidence=True
                    )
                else:
                    result = original_process_image(
                        temp_img_path, craft_net, reader, 
                        text_threshold, link_threshold, 
                        low_text, canvas_size, 
                        mag_ratio, poly, 
                        output_dir, paragraph, detail, allowlist,
                        direct_img=False, return_coords=False, return_confidence=False
                    )
                
                # 임시 파일 삭제
                try:
                    os.remove(temp_img_path)
                    print_log("임시 이미지 파일이 삭제되었습니다", "info")
                except:
                    print_log("임시 이미지 파일 삭제 실패", "warn")
                
                return result
                
            except Exception as e:
                print_log(f"이미지 처리 중 오류 발생: {e}", "error")
                if return_coords and return_confidence:
                    return None, [], [], [], [], []
                elif return_coords:
                    return None, [], [], [], []
                elif return_confidence:
                    return None, [], [], []
                else:
                    return None, [], []
        else:
            # 원본 함수 호출 (파일 경로 사용)
            return original_process_image(
                image_path_or_img, craft_net, reader, 
                text_threshold, link_threshold, 
                low_text, canvas_size, 
                mag_ratio, poly, 
                output_dir, paragraph, detail, allowlist,
                direct_img=False, return_coords=return_coords, return_confidence=return_confidence
            )
    
    # 함수 교체
    ocrtest.process_image = patched_process_image

# 메인 함수
def main():
    # 설정 로드
    # host = "127.0.0.1"
    host = "0.0.0.0"
    port = 8881
    craft_model_path = 'CRAFT-pytorch/weights/craft_mlt_25k.pth'
    lang_list = ['ko', 'en']
    lang_list = ['en']
    
    # process_image 패치 (직접 이미지 처리 지원)
    process_image_patch()
    
    # 서버 시작
    server = OCRServer(
        host=host, 
        port=port, 
        craft_model_path=craft_model_path,
        lang_list=lang_list
    )
    
    try:
        if server.start_server():
            server.accept_thread.join()
        else:
            print_log("서버를 시작할 수 없습니다.", "error")
    except KeyboardInterrupt:
        server.stop_server()
        print_log("키보드 인터럽트로 서버 종료", "info")
    except Exception as e:
        server.stop_server()
        print_log(f"오류로 서버 종료: {e}", "error")

if __name__ == "__main__":
    main()
