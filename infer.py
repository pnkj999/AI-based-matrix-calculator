import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

# --- Define Paths (User Configurable) ---
WEIGHT_PATH = '/home/msi/course/crohme/weights/trocr/trocr_crohme_best.pth'
VAL_LABEL_FILE = '/home/msi/course/crohme/CROHME23/TC11_CROHME23/labels/validation_labels/valid_labels.txt'  # Change this
VAL_IMAGE_DIR = '/home/msi/course/crohme/CROHME23/TC11_CROHME23/IMG/val/CROHME2016_test'        # Change this

# --- Load Model and Processor ---
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))  # Load trained weights
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = processor.tokenizer.vocab_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# --- Validation Loop ---
with open(VAL_LABEL_FILE, 'r') as f:
    lines = f.readlines()

correct_exact = 0
correct_token = 0
total_samples = 0
total_tokens = 0

for line in lines:
    if '\t' in line:
        img_path, gt_label = line.strip().split('\t', 1)
        full_img_path = os.path.join(VAL_IMAGE_DIR, os.path.basename(img_path))
        if not os.path.exists(full_img_path):
            print(f"Warning: Image not found: {full_img_path}")
            continue
        
        # Load and process image
        image = Image.open(full_img_path).convert('RGB')
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=128)
            pred_label = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Exact match accuracy
        if pred_label == gt_label:
            correct_exact += 1
        
        # Token-level accuracy
        pred_tokens = pred_label.split()
        gt_tokens = gt_label.split()
        min_len = min(len(pred_tokens), len(gt_tokens))
        matches = sum(1 for i in range(min_len) if pred_tokens[i] == gt_tokens[i])
        correct_token += matches
        total_tokens += max(len(pred_tokens), len(gt_tokens))
        
        total_samples += 1
        print(f'Predicted: "{pred_label}", Ground Truth: "{gt_label}"')
        
        # Clear GPU memory after each inference to avoid OOM
        torch.cuda.empty_cache()

# --- Compute and Print Accuracy ---
exact_accuracy = correct_exact / total_samples * 100 if total_samples > 0 else 0
token_accuracy = correct_token / total_tokens * 100 if total_tokens > 0 else 0

print(f'\nValidation Exact Match Accuracy: {exact_accuracy:.2f}% ({correct_exact}/{total_samples})')
print(f'Validation Token-Level Accuracy: {token_accuracy:.2f}% ({correct_token}/{total_tokens})')
