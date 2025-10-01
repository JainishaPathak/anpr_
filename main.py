import argparse
import cv2
import os
import re
from datetime import datetime
from collections import Counter, defaultdict
from ultralytics import YOLO
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class ImprovedANPR:
    VALID_STATE_CODES = {
        "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JK", "JH", "KA",
        "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "RJ", "SK", 
        "TN", "TS", "TR", "UP", "UK", "UT", "WB", "AN", "CH", "DH", "DD", "DL", 
        "LD", "PY"
    }

    def __init__(self, output_dir="output", use_easyocr=False):
        print("Loading models...")
        try:
            self.model = YOLO('weights/license_plate_detector.pt')
        except:
            self.model = YOLO('yolov8n.pt')
        
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        
        if torch.cuda.is_available():
            self.ocr_model.cuda()
        
        self.use_easyocr = use_easyocr
        self.easyocr_reader = None
        if use_easyocr:
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        
        self.output_dir = output_dir
        
        os.makedirs(f"{output_dir}/videos", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/detected_plates", exist_ok=True)
        
        self.vehicle_tracks = {}
        self.confirmed_plates = {}
        
        print("Ready\n")
    
    def fix_indian_plate(self, text):
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if len(text) < 9:
            return text
        
        result = list(text)
        
        for i in range(min(2, len(result))):
            if result[i].isdigit():
                char_map = {
                    '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', 
                    '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'
                }
                result[i] = char_map.get(result[i], result[i])
        
        for i in range(2, min(4, len(result))):
            if result[i].isalpha():
                char_map = {
                    'O': '0', 'D': '0', 'Q': '0',
                    'I': '1', 'L': '1', 'T': '1',
                    'Z': '2', 'B': '8', 'S': '5', 'G': '6'
                }
                result[i] = char_map.get(result[i], result[i])
        
        for i in range(4, min(6, len(result))):
            if result[i].isdigit():
                char_map = {
                    '0': 'O', '1': 'I', '2': 'Z', '3': 'B',
                    '4': 'A', '5': 'S', '6': 'G', '8': 'B'
                }
                result[i] = char_map.get(result[i], result[i])
        
        for i in range(6, len(result)):
            if result[i].isalpha():
                char_map = {
                    'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1',
                    'Z': '2', 'B': '8', 'S': '5', 'G': '6',
                    'M': '0', 'N': '0', 'H': '8', 'U': '0'
                }
                result[i] = char_map.get(result[i], result[i])
        
        return ''.join(result)
    
    def validate_indian_plate(self, text):
        if len(text) < 9 or len(text) > 10:
            return False
        
        if text[:2] not in self.VALID_STATE_CODES:
            return False
        
        patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$',
        ]
        
        return any(re.match(p, text) for p in patterns)
    
    def enhance_plate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        if h < 60:
            scale = 80 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        methods = []
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        methods.append(bilateral)
        
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append(thresh)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(otsu)
        
        return methods
    
    def read_plate_trocr(self, plate_img):
        all_readings = []
        
        def ocr_image(img):
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img).convert('RGB')
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
            
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            generated_ids = self.ocr_model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text
        
        try:
            text = ocr_image(plate_img)
            text = self.fix_indian_plate(text)
            if self.validate_indian_plate(text):
                all_readings.append(text)
        except:
            pass
        
        for enhanced in self.enhance_plate(plate_img):
            try:
                text = ocr_image(enhanced)
                text = self.fix_indian_plate(text)
                if self.validate_indian_plate(text):
                    all_readings.append(text)
            except:
                pass
        
        if all_readings:
            return Counter(all_readings).most_common(1)[0][0]
        
        return None
    
    def read_plate_easyocr(self, plate_img):
        if not self.easyocr_reader:
            return None
        
        all_readings = []
        
        try:
            results = self.easyocr_reader.readtext(plate_img, detail=0, paragraph=False)
            for text in results:
                cleaned = self.fix_indian_plate(text)
                if self.validate_indian_plate(cleaned):
                    all_readings.append(cleaned)
        except:
            pass
        
        for enhanced in self.enhance_plate(plate_img):
            try:
                results = self.easyocr_reader.readtext(enhanced, detail=0, paragraph=False)
                for text in results:
                    cleaned = self.fix_indian_plate(text)
                    if self.validate_indian_plate(cleaned):
                        all_readings.append(cleaned)
            except:
                pass
        
        if all_readings:
            return Counter(all_readings).most_common(1)[0][0]
        
        return None
    
    def read_plate(self, plate_img):
        result = self.read_plate_trocr(plate_img)
        
        if not result and self.use_easyocr:
            result = self.read_plate_easyocr(plate_img)
        
        return result
    
    def process_video(self, video_path, output_name=None, save_video=True):
        is_webcam = video_path.lower() in ['webcam', 'camera', '0']
        
        if is_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open {'webcam' if is_webcam else video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        
        if output_name is None:
            output_name = f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        out = None
        if save_video and not is_webcam:
            out_video = f"{self.output_dir}/videos/{output_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
        
        log_path = f"{self.output_dir}/logs/{output_name}.txt"
        log = open(log_path, 'w', encoding='utf-8')
        log.write(f"ANPR Tracking Log\n{'='*70}\n")
        log.write(f"Source: {'Webcam' if is_webcam else os.path.basename(video_path)}\n")
        log.write(f"Date: {datetime.now()}\n\n")
        
        frame_num = 0
        plate_count = 0
        paused = False
        
        print(f"Processing: {os.path.basename(video_path) if not is_webcam else 'Webcam'}")
        if not is_webcam:
            print(f"Frames: {total} | FPS: {fps:.1f}")
        print("-" * 70)
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        continue
                    else:
                        break
                
                timestamp = frame_num / fps
                display = frame.copy()
                
                if frame_num % 2 == 0:
                    results = self.model.track(frame, persist=True, conf=0.30, 
                                              verbose=False, tracker="bytetrack.yaml")
                    
                    for result in results:
                        if result.boxes is not None and result.boxes.id is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            track_ids = result.boxes.id.cpu().numpy().astype(int)
                            
                            for box, track_id in zip(boxes, track_ids):
                                x1, y1, x2, y2 = map(int, box)
                                
                                if track_id not in self.vehicle_tracks:
                                    self.vehicle_tracks[track_id] = {
                                        'readings': [],
                                        'confirmed': None,
                                        'logged': False,
                                        'box': (x1, y1, x2, y2),
                                        'last_updated': frame_num
                                    }
                                
                                self.vehicle_tracks[track_id]['box'] = (x1, y1, x2, y2)
                                self.vehicle_tracks[track_id]['last_updated'] = frame_num
                                
                                pad = 20
                                x1_crop = max(0, x1 - pad)
                                y1_crop = max(0, y1 - pad)
                                x2_crop = min(width, x2 + pad)
                                y2_crop = min(height, y2 + pad)
                                
                                plate_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                                
                                if plate_img.size > 0 and plate_img.shape[0] > 20:
                                    plate_text = self.read_plate(plate_img)
                                    
                                    if plate_text:
                                        self.vehicle_tracks[track_id]['readings'].append(plate_text)
                                        
                                        if len(self.vehicle_tracks[track_id]['readings']) >= 3:
                                            readings = self.vehicle_tracks[track_id]['readings']
                                            counter = Counter(readings)
                                            most_common, count = counter.most_common(1)[0]
                                            
                                            if count >= 3 and not self.vehicle_tracks[track_id]['confirmed']:
                                                self.vehicle_tracks[track_id]['confirmed'] = most_common
                                                self.confirmed_plates[track_id] = most_common
                                                
                                                plate_count += 1
                                                crop_path = f"{self.output_dir}/detected_plates/plate_{plate_count}_T{track_id}_{most_common}.jpg"
                                                cv2.imwrite(crop_path, plate_img)
                                                
                                                if not self.vehicle_tracks[track_id]['logged']:
                                                    entry = f"[{timestamp:7.2f}s] Track {track_id}: {most_common}\n"
                                                    log.write(entry)
                                                    log.flush()
                                                    print(f"[{timestamp:7.2f}s] Track {track_id}: {most_common}")
                                                    self.vehicle_tracks[track_id]['logged'] = True
                
                for track_id, data in list(self.vehicle_tracks.items()):
                    if data['confirmed']:
                        if (frame_num - data['last_updated']) > 150:
                            continue
                        
                        if (frame_num - data['last_updated']) <= 5:
                            x1, y1, x2, y2 = data['box']
                            
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            label = data['confirmed']
                            
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(display, (x1, y1-th-12), (x1+tw+10, y1), (0, 255, 0), -1)
                            cv2.putText(display, label, (x1+5, y1-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                cv2.rectangle(display, (5, 5), (420, 100), (0, 0, 0), -1)
                cv2.putText(display, f"Time: {timestamp:.2f}s", (15, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if is_webcam:
                    cv2.putText(display, f"Frame: {frame_num}", (15, 58),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(display, f"Frame: {frame_num}/{total}", (15, 58),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display, f"Detections: {len(self.confirmed_plates)}", (15, 86),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if save_video and out:
                    out.write(display)
                
                frame_num += 1
            
            cv2.imshow('ANPR Detection', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s'):
                screenshot_path = f"{self.output_dir}/screenshot_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, display)
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        log.write(f"\n{'='*70}\n")
        log.write(f"Total detections: {len(self.confirmed_plates)}\n\n")
        
        log.write(f"Detected plates:\n")
        for tid, plate in sorted(self.confirmed_plates.items()):
            log.write(f"  Track {tid}: {plate}\n")
        log.close()
        
        print(f"\n{'='*70}")
        print("Processing complete")
        if save_video and not is_webcam:
            print(f"Video: {out_video}")
        print(f"Log: {log_path}")
        print(f"Total detections: {len(self.confirmed_plates)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="tracked")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--use-easyocr", action="store_true")
    args = parser.parse_args()
    
    anpr = ImprovedANPR(use_easyocr=args.use_easyocr)
    anpr.process_video(args.video, args.output, save_video=not args.no_save)

if __name__ == "__main__":
    main()