import cv2
import os
from PIL import Image

def extract_frames_from_gif(gif_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gif = Image.open(gif_path)
    frame_index = 0
    try:
        while True:
            frame = gif.convert('RGB')
            frame.save(f"{output_dir}/frame_{frame_index:03d}.jpg", 'JPEG')
            frame_index += 1
            gif.seek(gif.tell() + 1)
    except EOFError:
        print(f"[INFO] Extracted {frame_index} frames from {gif_path}")

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
