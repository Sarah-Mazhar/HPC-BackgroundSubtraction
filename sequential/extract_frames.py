import cv2
import os

def extract_frames_from_gif(gif_path, output_folder):
    cap = cv2.VideoCapture(gif_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, f'frame_{i:03d}.jpg'), gray)
        i += 1
    cap.release()
    print(f"✅ Extracted {i} frames to '{output_folder}'")

if __name__ == "__main__":
    os.makedirs("sequential/extracted_frames", exist_ok=True)
    extract_frames_from_gif("sequential/input/Background-Subtraction-Tutorial_merged.gif", "sequential/extracted_frames")
