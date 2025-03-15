# matrix-calculator
We have created a tool for basic  real matrix opertions like addition , subtraction , multiplication, transpose, inverse, Eigen value ,Eigen vectors and Trace calculation where a user can give input in two forms 
1) User can draw a matrix manually then take latex code from website and paste in our GUI and then tell our GUI which operations he want to perform based on above and then it will give result based on the input and operation
2) Here user can enter matrix from his/her keyboard and then tell gui which operation he want.
3) This option is not added yet where User can input image of matrix and it performs required operation as matrix images are very less in internet hence our model was giving very less accuracy so we have to remove this option as augmentation will not work in this case.





We have provided .py file for our code wihch we have merged with streamlit for webpage application


For updated code these operations will come 
"""
    **Enter a LaTeX matrix operation**  
    *Supports:*  
    - **Transpose:** Tr =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Trace:** Trc =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Inverse:** inv= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Determinant:** det= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Eigenvalues:** E =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Eigenvectors:** X= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Dimension (Rank):** dim =\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
    - **Rank:** R =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **LU Decomposition:** LU =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **SVD Decomposition:** SVD =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Matrix Addition, Subtraction, Multiplication (supports multiple matrices)**  
      Example:  

\begin{bmatrix} 1 & 2 \end{bmatrix} + \begin{bmatrix} 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \end{bmatrix}

    """



    NOw we have traine our model on tr ocr model and you can see from hee the model rquriements 

    TrOCR Training on CROHME Dataset
    Link for weight files
https://drive.google.com/file/d/1GE3P-dh6in6g84fxJfCbet4UiNI6KJZU/view?usp=sharing

This repository contains code to train a TrOCR model (`microsoft/trocr-small-handwritten`) on the CROHME dataset for handwritten mathematical expression recognition. The training process uses a custom dataset class, mixed precision (FP16), and gradient accumulation to fit a small GPU (3.6 GiB).

Dataset
- Source: CROHME 2023 and 2016 combined  dataset (`TC11_CROHME23`).
- Training Data:
  - Labels: `/home/msi/course/crohme/CROHME23/TC11_CROHME23/labels/combined_labels.txt` (15,523 samples).
  - Images: `/home/msi/course/crohme/CROHME23/TC11_CROHME23/IMG/combined_images/`.
- Format: Tab-separated file with image paths and LaTeX labels (e.g., `image.png\t$x + 2$`).

Model
- Base Model: `microsoft/trocr-small-handwritten` (~60M parameters).
- Why Small?: Chosen to fit a 3.6 GiB GPU, unlike `trocr-base-handwritten` (~330M), which caused OOM errors.

Training Setup
- Hardware: GPU with 3.6 GiB VRAM (e.g., NVIDIA GTX 1650).
- Environment: Conda (`C2F-Seg`), Python 3.10.
- Dependencies:
  pip install transformers torch torchvision Pillow sentencepiece protobuf
- Script: `train_trocr.py`.

Key Features
- Batch Size: 1 (due to memory constraints).
- Gradient Accumulation: 4 steps, simulating an effective batch size of 4.
- Mixed Precision: FP16 via `torch.cuda.amp` to reduce memory usage.
- Optimizer: AdamW, learning rate 5e-5.
- Scheduler: `ReduceLROnPlateau`, patience 3, factor 0.5.
- Epochs: 20.
- Output: Weights saved to `/home/msi/course/crohme/weights/trocr/trocr_crohme_best.pth`.

How I Trained It
1. Initial Attempt:
   - Tried `trocr-base-handwritten` with batch size 4.
   - Failed with `CUDA out of memory` (3.6 GiB GPU limit).
2. Adjustments:
   - Switched to `trocr-small-handwritten`.
   - Reduced batch size to 1.
   - Added FP16 training.
   - Used gradient accumulation (4 steps).
3. Dependency Issues:
   - `protobuf` missing: Fixed with `pip install protobuf`.
   - `sentencepiece` missing: Fixed with `pip install sentencepiece`.
   - Tokenizer errors: Set `use_fast=False` in `TrOCRProcessor`.
4. Final Run:
   - Trained for 20 epochs on 15,523 samples.
   - Final loss: 0.0190 (Epoch 20).
   - Weights saved when loss improved.

Training Script (`train_trocr.py`)

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# ... (rest of imports and code)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
# ... (rest of setup and training loop)

- Run Command:
  python train_trocr.py

Results
- Training Loss: Dropped to 0.0190 after 20 epochs.
- Output: Best weights at `/home/msi/course/crohme/weights/trocr/trocr_crohme_best.pth`.

Validation
- Use `infer_trocr.py` (not included here) to check accuracy:
  - Update paths for validation labels and images.
  - Run: `python infer_trocr.py`.
- Metrics: Exact match and token-level accuracy.

How to Reproduce
1. Setup Environment:
   conda create -n C2F-Seg python=3.10
   conda activate C2F-Seg
   pip install transformers torch torchvision Pillow sentencepiece protobuf
2. Prepare Data:
   - Place labels in `/path/to/labels/combined_labels.txt`.
   - Place images in `/path/to/images/combined_images/`.
3. Update Paths:
   - Edit `LABEL_FILE` and `IMAGE_DIR` in `train_trocr.py`.
4. Train:
   python train_trocr.py

Notes
- GPU Memory: If OOM occurs, clear cache (`torch.cuda.empty_cache()`) or reduce `accumulation_steps`.
- Time: ~4–8 hours/epoch on a 3.6 GiB GPU with 15,523 samples.
- Improvements: Could increase epochs or fine-tune hyperparameters for better accuracy.

Acknowledgments
- Uses Hugging Face’s `transformers` library.

