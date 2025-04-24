# Parallel Background Subtraction using MPI (Python)

## 📌 Project Description

This project implements **Parallel Background Subtraction** using the **Message Passing Interface (MPI)** with Python and `mpi4py`.  
The input is a GIF video from which frames are extracted and processed in parallel across multiple processes. The goal is to:

- Estimate the **background image** (in grayscale).
- Generate **foreground masks** (binary images showing moving objects).
- Compile the foreground masks into a video (`foreground_mask.mp4`).

---

## 🎥 **Input and Output**

- **Input:**  
  `Background-Subtraction-Tutorial_merged.gif`

- **Output:**  
  - `output/estimated_background.jpg`: The estimated background (black and white).
  - `output/foreground/`: Foreground mask images (binary masks).
  - `output/foreground_mask.mp4`: Compiled video of the foreground masks.

---

## 🖥️ **Technologies Used**
- Python 3.x
- MPI via `mpi4py`
- OpenCV (`opencv-python`)
- NumPy
- Pillow (`PIL`)

---

## 🛠️ **Installation**

1. Clone or download this project.
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate

Perfect! Here’s your updated **`README.md`** with the run command included clearly under its own section ✅:

---

## 🚀 **Running the Project**

1. Make sure your input GIF is named:
   ```
   Background-Subtraction-Tutorial_merged.gif
   ```
   and is placed in the root folder.

2. Run the project using the following command:
```bash
mpirun -np 4 python3 main.py
```

> 🟢 **Note:**  
> Replace `4` with the number of processes (MPI workers) you want to use depending on your CPU cores.
