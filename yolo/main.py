import time
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    weights_path = Path(__file__).resolve().parent / "runs" / "detect" / "figures" / "yolo3" / "weights" / "best.pt"
    object_detector = YOLO(str(weights_path))
    
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Real-time Detection", cv2.WINDOW_GUI_NORMAL)
    
    is_detecting = True
    
    while camera.isOpened():
        success, current_frame = camera.read()
        if not success:
            print("Не удалось получить кадр с камеры.")
            break
            
        pressed_key = cv2.waitKey(1) & 0xFF
        
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('p'):
            is_detecting = not is_detecting
            status = 'Включена' if is_detecting else 'Выключена'
            print(f"Детекция: {status}")

        if is_detecting:
            start_time = time.time()
            predictions = object_detector.predict(current_frame, conf=0.4, verbose=False)
            inference_time = time.time() - start_time
            
            for res in predictions:
                for bbox in res.boxes:
                    xmin, ymin, xmax, ymax = [int(val) for val in bbox.xyxy[0]]
                    
                    score = float(bbox.conf[0])
                    class_id = int(bbox.cls[0])
                    class_name = object_detector.names[class_id]

                    box_color = (255, 255, 0)
                    cv2.rectangle(current_frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    
                    display_text = f"{class_name.upper()}: {score*100:.0f}%"
                    cv2.putText(current_frame, display_text, (xmin, max(15, ymin - 10)), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, box_color, 1)
                    
                    print(f"[{inference_time:.3f} сек] Объект: {class_name} | Уверенность: {score:.2f}")

        cv2.imshow("Real-time Detection", current_frame)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()