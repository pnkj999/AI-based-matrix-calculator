import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
from torch.cuda.amp import GradScaler, autocast

# Set environment variable to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Define Paths (User Configurable) ---
LABEL_FILE = '/home/msi/course/crohme/CROHME23/TC11_CROHME23/labels/combined_labels.txt'  # Change this
IMAGE_DIR = '/home/msi/course/crohme/CROHME23/TC11_CROHME23/IMG/combined_images/'         # Change this
WEIGHT_SAVE_DIR = '/home/msi/course/crohme/weights/trocr/'
SAVE_PATH = os.path.join(WEIGHT_SAVE_DIR, 'trocr_crohme_best.pth')
CHECKPOINT_PATH = os.path.join(WEIGHT_SAVE_DIR, 'trocr_crohme_checkpoint.pth')
os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)

# --- Dataset Class ---
class CROHMEDataset(Dataset):
    def __init__(self, label_file, image_dir, processor, max_target_length=128):
        self.image_dir = image_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.image_paths = []
        self.labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                if '\t' in line:
                    img_path, label = line.strip().split('\t', 1)
                    full_img_path = os.path.join(self.image_dir, os.path.basename(img_path))
                    if not os.path.exists(full_img_path):
                        print(f"Warning: Image not found: {full_img_path}")
                        continue
                    self.image_paths.append(full_img_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)  # [C, H, W]
        
        encoding = self.processor.tokenizer(label, 
                                          padding="max_length", 
                                          max_length=self.max_target_length, 
                                          truncation=True, 
                                          return_tensors="pt")
        labels = encoding.input_ids.squeeze(0)  # [max_target_length]
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# --- Training Setup ---
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)  # Use slow tokenizer
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = processor.tokenizer.vocab_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

dataset = CROHMEDataset(LABEL_FILE, IMAGE_DIR, processor)
print(f"Dataset size: {len(dataset)}")
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
scaler = GradScaler()

# --- Training Loop ---
num_epochs = 20
best_loss = float('inf')
accumulation_steps = 4  # Effective batch size = 4

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        batch_count += 1
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    avg_epoch_loss = epoch_loss / batch_count
    print(f'Epoch {epoch+1}/{num_epochs} Complete, Average Loss: {avg_epoch_loss:.4f}')
    scheduler.step(avg_epoch_loss)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f'Saved best weights to {SAVE_PATH} with loss {best_loss:.4f}')

print("Training complete.")
