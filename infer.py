import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

# --- Define Paths (User Configurable) ---
WEIGHT_PATH = '/home/msi/course/crohme/weights/trocr/trocr_crohme_best.pth'
IMAGE_PATH = '/home/msi/Downloads/form_5_657_E3285.png'  # Change this to your image path

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

# --- Prediction for Single Image ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
else:
    # Load and process image
    image = Image.open(IMAGE_PATH).convert('RGB')
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # Generate prediction
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128)
        pred_label = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f'Predicted Label: "{pred_label}"')
    
    # Clear GPU memory
    torch.cuda.empty_cache()
