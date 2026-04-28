import cv2, numpy as np, time, threading, ctypes
from ultralytics import YOLO
from pathlib import Path

def play_sound(path):
    ctypes.windll.winmm.mciSendStringW(f'close s', 0, 0, 0)
    ctypes.windll.winmm.mciSendStringW(f'open "{path}" alias s', 0, 0, 0)
    ctypes.windll.winmm.mciSendStringW('play s from 0', 0, 0, 0)

def get_angle(a, b, c):
    ba, bc = a - b, c - b
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1)))

def main():
    base_dir = Path(__file__).parent
    print("Loading model...")
    model = YOLO("yolo26n-pose.pt")
    
    video_path = str(base_dir / "test.mp4")
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    sound = str((base_dir / "sound.mp3").absolute())
    count, state, last_seen = 0, "UP", time.time()
    
    print("Press 'q' to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if f_idx % 30 == 0:
            print(f"Processing frame {f_idx}...")

        res = model(frame, verbose=True)[0]
        if res.keypoints and len(res.keypoints.data) > 0:
            last_seen = time.time()
            pts = res.keypoints.data[0].cpu().numpy()
            
            idx = [5, 7, 9] if pts[5, 2] > pts[6, 2] else [6, 8, 10]
            if all(pts[idx, 2] > 0.5):
                angle = get_angle(pts[idx[0], :2], pts[idx[1], :2], pts[idx[2], :2])
                
                if angle < 100 and state == "UP": state = "DOWN"
                if angle > 160 and state == "DOWN":
                    state = "UP"
                    count += 1
                    print(f"Pushup count: {count}")
                    threading.Thread(target=play_sound, args=(sound,), daemon=True).start()
                
                cv2.putText(frame, f"Angle: {int(angle)}", (30, 100), 0, 0.8, (255, 255, 255), 2)
            frame = res.plot()

        if time.time() - last_seen > 5: count = 0
    
        cv2.rectangle(frame, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"PUSHUPS: {count}", (20, 50), 0, 1, (0, 255, 0), 2)
        cv2.imshow("Pushup Counter", frame)
        
        if cv2.waitKey(1) == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
