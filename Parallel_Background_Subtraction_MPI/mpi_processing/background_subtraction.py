import cv2
import numpy as np
import os
from utils.image_utils import load_images, save_image, make_video_from_images

def split_list(data, num_chunks):
    avg = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
    return chunks

def estimate_background(images):
    stack = np.stack([img for img, _ in images], axis=3)
    background = np.median(stack, axis=3).astype(np.uint8)
    return background

def parallel_background_subtraction(comm, rank, size):
    if rank == 0:
        images = load_images("input/")
        background = estimate_background(images)

        # 🟠 ✅ Convert background to grayscale here (your requested fix):
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        save_image(background_gray, "output/estimated_background.jpg")  # Save as black & white (grayscale)

        chunks = split_list(images, size)
    else:
        chunks = None
        background = None

    # Broadcast the original color background (needed for foreground subtraction):
    background = comm.bcast(background, root=0)
    local_images = comm.scatter(chunks, root=0)

    processed_images = []
    for img, filename in local_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background_gray, gray)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        processed_images.append((mask, filename))

    gathered_results = comm.gather(processed_images, root=0)

    if rank == 0:
        if not os.path.exists("output/foreground"):
            os.makedirs("output/foreground")
        for process_results in gathered_results:
            for mask_img, filename in process_results:
                save_image(mask_img, os.path.join("output/foreground", filename))
        make_video_from_images("output/foreground", "output/foreground_mask.mp4")
