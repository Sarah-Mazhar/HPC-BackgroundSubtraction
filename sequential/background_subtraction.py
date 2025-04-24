import cv2
import numpy as np
import os
import imageio.v2 as imageio  # For GIF creation


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
    # Compute average background from stacked frames
    return np.mean(images, axis=0).astype(np.uint8)

# def subtract_background(background, test_frame, threshold=45):
#     # Subtract background and apply binary threshold
#     diff = cv2.absdiff(background, test_frame)
#     _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

#     # Apply morphological operations to remove noise and close holes
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small white noise
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill gaps in white blobs

#     return mask

def subtract_background(background, test_frame, threshold=45):
    diff = cv2.absdiff(background, test_frame)

    # Binary thresholding
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological filtering
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps in moving objects

    # Optional smoothing
    mask = cv2.medianBlur(mask, 5)

    return mask

def create_gif_from_frames(frame_folder, output_path, prefix="foreground_mask_", duration=0.1):
    images = []
    filenames = sorted([f for f in os.listdir(frame_folder) if f.startswith(prefix) and f.endswith('.jpg')])
    for filename in filenames:
        img = imageio.imread(os.path.join(frame_folder, filename))
        images.append(img)
    imageio.mimsave(output_path, images, duration=duration)
    print(f"🎞️ GIF created: {output_path}")



# def main():
#     frames_folder = "sequential/extracted_frames"
#     output_folder = "sequential/output"
#     os.makedirs(output_folder, exist_ok=True)

#     print("🔍 Loading background frames...")
#     images = load_images_from_folder(frames_folder, limit=50)  # Use more frames for better average
#     if len(images) == 0:
#         print("❌ No images found in folder.")
#         return

#     print("🧠 Computing background...")
#     background = compute_background(images)
#     cv2.imwrite(os.path.join(output_folder, "background.jpg"), background)

#     test_frame_path = os.path.join(frames_folder, "frame_070.jpg")  # choose any mid-frame
#     test_frame = cv2.imread(test_frame_path, cv2.IMREAD_GRAYSCALE)
#     if test_frame is None:
#         print(f"❌ Test frame {test_frame_path} not found!")
#         return
#     cv2.imwrite(os.path.join(output_folder, "test_frame.jpg"), test_frame)

#     print("📤 Performing background subtraction...")
#     mask = subtract_background(background, test_frame, threshold=45)
#     cv2.imwrite(os.path.join(output_folder, "foreground_mask.jpg"), mask)

#     print("✅ Done! Results saved in 'sequential/output/'")

def main():
    frames_folder = "sequential/extracted_frames"
    output_folder = "sequential/output"
    os.makedirs(output_folder, exist_ok=True)

    print("🔍 Loading background frames...")
    images = load_images_from_folder(frames_folder, limit=50)
    if len(images) == 0:
        print("❌ No images found in folder.")
        return

    print("🧠 Computing background...")
    background = compute_background(images)
    cv2.imwrite(os.path.join(output_folder, "background.jpg"), background)

    print("📤 Performing background subtraction for multiple test frames...")
    for i in range(60, 80):  # Adjust range to fit your video
        frame_name = f"frame_{i:03d}.jpg"
        test_frame_path = os.path.join(frames_folder, frame_name)
        test_frame = cv2.imread(test_frame_path, cv2.IMREAD_GRAYSCALE)
        if test_frame is None:
            continue
        mask = subtract_background(background, test_frame, threshold=45)

        # Save frame and mask
        cv2.imwrite(os.path.join(output_folder, f"foreground_mask_{i}.jpg"), mask)

    # 🎞️ Create GIF from saved mask frames
    create_gif_from_frames(output_folder, os.path.join(output_folder, "foreground_mask.gif"))

    print("✅ All done! Output and GIF saved in 'sequential/output/'")


if __name__ == "__main__":
    main()
