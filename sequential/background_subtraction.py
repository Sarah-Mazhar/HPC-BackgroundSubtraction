import cv2
import numpy as np
import os

def load_images_from_folder(folder_path, limit=None):
    images = []
    filenames = sorted(os.listdir(folder_path))
    if limit:
        filenames = filenames[:limit]
    for filename in filenames:
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return np.array(images)

def compute_background(images):
    return np.mean(images, axis=0).astype(np.uint8)

def subtract_background(background, test_frame, threshold=30):
    diff = cv2.absdiff(background, test_frame)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask

def main():
    frames_folder = "sequential/extracted_frames"
    output_folder = "sequential/output"
    os.makedirs(output_folder, exist_ok=True)

    print("🔍 Loading background frames...")
    images = load_images_from_folder(frames_folder, limit=10)  # average of first 10 frames
    if len(images) == 0:
        print("❌ No images found in folder.")
        return

    print("🧠 Computing background...")
    background = compute_background(images)
    cv2.imwrite(os.path.join(output_folder, "background.jpg"), background)

    test_frame_path = os.path.join(frames_folder, "frame_015.jpg")  # you can change the test frame index
    test_frame = cv2.imread(test_frame_path, cv2.IMREAD_GRAYSCALE)
    if test_frame is None:
        print(f"❌ Test frame {test_frame_path} not found!")
        return
    cv2.imwrite(os.path.join(output_folder, "test_frame.jpg"), test_frame)

    print("📤 Performing background subtraction...")
    mask = subtract_background(background, test_frame, threshold=30)
    cv2.imwrite(os.path.join(output_folder, "foreground_mask.jpg"), mask)

    print("✅ Done! Results saved in 'sequential/output/'")

if __name__ == "__main__":
    main()
