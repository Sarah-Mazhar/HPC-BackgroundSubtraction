import cv2
import os

def extract_frames_from_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{output_dir}/frame_{frame_index:03d}.jpg", frame_rgb)
        frame_index += 1

    cap.release()
    print(f"[INFO] Extracted {frame_index} frames from {video_path}")

def load_images(input_dir):
    images = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_dir, filename))
            images.append((img, filename))
    return images

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def make_video_from_images(image_folder, output_video_path, fps=10):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        print("No images found for video creation.")
        return
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image), cv2.IMREAD_GRAYSCALE)
        video.write(frame)
    video.release()
    print(f"[INFO] Video saved to {output_video_path}")
