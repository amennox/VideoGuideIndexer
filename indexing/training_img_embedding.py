import requests
import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import open_clip
import ssl
import shutil
import time
import logging

ssl._create_default_https_context = ssl._create_unverified_context

log = logging.getLogger("indexing.training_img_embedding")
log.setLevel(logging.INFO)

def fine_tune_openclip_from_ftimages(scope, stream_callback):
    """
    Fine-tunes an open_clip model using images and descriptions from a remote FtImages API filtered by scope.
    The stream_callback(progress:int, stage:str, info:str) is called to update progress to the frontend.
    """
    FTIMAGES_API_URL = f"http://localhost:5209/FtImages"
    MODEL_NAME = "ViT-B-32"
    LOCAL_MODEL_PATH = "./models/vit_b_32/open_clip_pytorch_model.bin"
    OUTPUT_MODEL_PATH = f"./ft_images_{scope}.pth"
    TRAIN_DATA_PATH = f"./frames/training_dataset_{scope}.json"
    IMAGES_DIR = Path(f"./frames/ftimages_training_{scope}")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"[START] Fine-tuning OpenCLIP for scope '{scope}'")
    stream_callback(0, "fetching_data", "Fetching FtImages data...")
    params = {"scope": scope, "page": 1, "pageSize": 1000}
    try:
        resp = requests.get(FTIMAGES_API_URL, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Error fetching FtImages API: {e}")
        stream_callback(0, "error", f"Error fetching FtImages API: {e}")
        return
    items = resp.json().get("items", [])
    if not items:
        log.warning(f"No items found for scope '{scope}'")
        stream_callback(0, "error", "No items found for this scope.")
        return

    log.info(f"Found {len(items)} items for scope '{scope}'")
    data = []
    errors = 0
    for idx, item in enumerate(items):
        image_url = item.get("imageUrl")
        image_id = item.get("id")
        description = item.get("description", "")
        if not image_url:
            log.warning(f"[{idx}] Item {image_id}: missing imageUrl")
            errors += 1
            continue

        img_ext = os.path.splitext(image_url)[-1]
        if img_ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            img_ext = ".jpg"
        img_local_path = IMAGES_DIR / f"{image_id}{img_ext}"
        # Download image if not present
        if not img_local_path.exists():
            try:
                log.info(f"[{idx+1}/{len(items)}] Downloading {image_url} to {img_local_path}")
                img_resp = requests.get(image_url, timeout=30)
                if img_resp.status_code != 200 or not img_resp.content:
                    raise Exception(f"HTTP {img_resp.status_code}, empty content")
                with open(img_local_path, "wb") as f:
                    f.write(img_resp.content)
                # Verify the image is valid and can be opened
                try:
                    _ = Image.open(img_local_path).verify()
                except Exception as e:
                    log.error(f"Downloaded file is not a valid image: {img_local_path}")
                    img_local_path.unlink(missing_ok=True)
                    raise Exception(f"Not a valid image: {e}")
            except Exception as e:
                log.error(f"Error downloading {image_url}: {e}")
                stream_callback(int(5 * idx / max(1, len(items))), "download_error", f"Error downloading {image_url}: {e}")
                errors += 1
                continue

        data.append({
            "image_path": str(img_local_path),
            "description": description,
        })
        # Progress every 5 images or last image
        if idx % 5 == 0 or idx == len(items) - 1:
            stream_callback(int(5 + 10 * (idx+1) / len(items)), "downloading_images", f"Downloaded {idx+1}/{len(items)} images")

    if not data:
        log.error("No images downloaded successfully. Cannot continue training.")
        stream_callback(0, "error", "No images downloaded successfully.")
        return

    log.info(f"{len(data)} images ready for training (errors: {errors})")
    with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    stream_callback(20, "preparing_model", "Preparing model and dataset...")

    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=LOCAL_MODEL_PATH
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    class UIDataset(Dataset):
        def __init__(self, json_path, transform):
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            img_path = item["image_path"]
            description = item["description"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                log.error(f"Error opening image {img_path}: {e}")
                raise
            img_tensor = self.transform(img)
            return img_tensor, description

    dataset = UIDataset(TRAIN_DATA_PATH, preprocess_train)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    best_loss = float("inf")
    EPOCHS = 10
    total_steps = EPOCHS * len(dataloader)
    global_step = 0

    log.info(f"Start training loop, dataset size: {len(dataset)}, batch size: 4, epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (imgs, texts) in enumerate(dataloader):
            texts_tok = tokenizer(texts).to(device)
            imgs = imgs.to(device)
            optimizer.zero_grad()
            image_features, text_features, logit_scale = model(imgs, texts_tok)
            logit_scale_clamped = torch.clamp(logit_scale, 0, 4.6052)
            logits_per_image = image_features @ text_features.t() * logit_scale_clamped.exp()
            labels = torch.arange(len(imgs), device=device)
            loss = (
                torch.nn.functional.cross_entropy(logits_per_image, labels)
                + torch.nn.functional.cross_entropy(logits_per_image.t(), labels)
            ) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            global_step += 1
            percent = int(20 + (global_step / total_steps) * 79)
            if i % 2 == 0 or i == len(dataloader) - 1:
                log.info(f"Epoch {epoch+1}/{EPOCHS} Batch {i+1}/{len(dataloader)} Loss {loss.item():.4f}")
            stream_callback(percent, "training", f"Epoch {epoch+1}/{EPOCHS} - Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        loss_avg = total_loss / len(dataloader)
        stream_callback(percent, "epoch_done", f"Epoch {epoch+1} completed - Avg loss: {loss_avg:.6f}")
        log.info(f"Epoch {epoch+1} completed. Avg loss: {loss_avg:.6f}")
        if loss_avg < best_loss:
            best_loss = loss_avg
            torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
            log.info(f"Saved new best model to {OUTPUT_MODEL_PATH} (loss {loss_avg:.6f})")
            stream_callback(percent, "saving_model", f"Saved best model at epoch {epoch+1} with avg loss {loss_avg:.6f}")

    log.info(f"Training finished. Best loss: {best_loss:.6f}. Model: {OUTPUT_MODEL_PATH}")
    stream_callback(100, "done", f"Training finished. Best model saved as {OUTPUT_MODEL_PATH}")
    # Optional: clean up images (uncomment if desired)
    # shutil.rmtree(IMAGES_DIR)
