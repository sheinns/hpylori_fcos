import os
import glob
import random
from PIL import Image

# Configuration
patch_size = 512
stride = 256  # 50% overlap — recovers boundary objects and ~doubles dataset size
images_dir = "data_2006/dataset/positive/images"
pascal_dir = "data_2006/dataset/positive/pascal"
neg_dir = "data_2006/dataset/negative"

out_images_dir = "data_2006/patches/images"
out_pascal_dir = "data_2006/patches/pascal"


os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_pascal_dir, exist_ok=True)

# To prevent overwhelming the dataset with empty patches, we randomly keep a fraction of them.
# With stride=256, mixed-size images (1920x1080 to 3840x2160) produce varying patch counts.
empty_patch_keep_prob = 0.01

total_pos_patches = 0
total_neg_patches = 0
total_skipped_empty = 0

# Minimum box area (in pixels) to keep after clipping to patch — filters degenerate slivers
MIN_BOX_AREA = 50

def process_image(img_path, label_path=None, is_negative_image=False):
    global total_pos_patches, total_neg_patches, total_skipped_empty
    img_name = os.path.basename(img_path)
    
    # Read original boxes if they exist
    boxes = []
    if not is_negative_image and label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xmin, ymin, xmax, ymax = map(float, parts)
                    boxes.append([c, xmin, ymin, xmax, ymax])
                    
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {img_path}: {e}")
        return
        
    w, h = img.size
    
    # Slide window
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_left, y_top = x, y
            x_right, y_bottom = min(x + patch_size, w), min(y + patch_size, h)
            
            # Skip incomplete patches at the edges
            if (x_right - x_left) < patch_size or (y_bottom - y_top) < patch_size:
                continue
                
            patch_boxes = []
            for box in boxes:
                c, b_xmin, b_ymin, b_xmax, b_ymax = box
                b_xcenter = (b_xmin + b_xmax) / 2.0
                b_ycenter = (b_ymin + b_ymax) / 2.0
                
                # Check if box center is inside the patch
                if x_left <= b_xcenter <= x_right and y_top <= b_ycenter <= y_bottom:
                    new_xmin = max(0, b_xmin - x_left)
                    new_ymin = max(0, b_ymin - y_top)
                    new_xmax = min(patch_size, b_xmax - x_left)
                    new_ymax = min(patch_size, b_ymax - y_top)
                    
                    box_w = new_xmax - new_xmin
                    box_h = new_ymax - new_ymin
                    
                    # Filter out degenerate slivers from clipping
                    if box_w > 3 and box_h > 3 and (box_w * box_h) >= MIN_BOX_AREA:
                        patch_boxes.append(f"{int(c)} {new_xmin:.2f} {new_ymin:.2f} {new_xmax:.2f} {new_ymax:.2f}\n")
            
            is_empty = len(patch_boxes) == 0
            
            # Save the patch if it has objects OR (it's empty AND it passes our probability check)
            if not is_empty:
                patch_id = f"{img_name.replace('.jpg', '').replace('.png', '')}_y{y}_x{x}"
                patch_img_name = f"{patch_id}.png"
                patch_label_name = f"{patch_id}.txt"
                
                patch_img = img.crop((x_left, y_top, x_right, y_bottom))
                patch_img.save(os.path.join(out_images_dir, patch_img_name))
                
                with open(os.path.join(out_pascal_dir, patch_label_name), "w") as out_f:
                    out_f.writelines(patch_boxes)
                    
                total_pos_patches += 1
            elif random.random() < empty_patch_keep_prob:
                patch_id = f"{img_name.replace('.jpg', '').replace('.png', '')}_y{y}_x{x}"
                patch_img_name = f"{patch_id}.png"
                patch_label_name = f"{patch_id}.txt"
                
                patch_img = img.crop((x_left, y_top, x_right, y_bottom))
                patch_img.save(os.path.join(out_images_dir, patch_img_name))
                
                # Write empty label file
                with open(os.path.join(out_pascal_dir, patch_label_name), "w") as out_f:
                    pass
                    
                total_neg_patches += 1
            else:
                total_skipped_empty += 1

# 1. Process Positive Images
print("Processing positive images...")
pos_image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
for img_path in pos_image_paths:
    label_path = os.path.join(pascal_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
    process_image(img_path, label_path=label_path, is_negative_image=False)

# 2. Process Negative Images
print("Processing negative images...")
neg_image_paths = glob.glob(os.path.join(neg_dir, "*.jpg"))
for img_path in neg_image_paths:
    process_image(img_path, is_negative_image=True)

print(f"\nTraining patchification complete!")
print(f"Positive patches (with H. Pylori): {total_pos_patches}")
print(f"Negative patches (pure background): {total_neg_patches}")
print(f"Skipped empty patches: {total_skipped_empty}")
print(f"Total patches saved: {total_pos_patches + total_neg_patches}")

# 3. Process Test Images into separate directory
print("\n--- Processing test set ---")
test_images_dir = "data_2006/dataset-test/images"
test_pascal_dir = "data_2006/dataset-test/pascal"
out_test_images_dir = "data_2006/test_patches/images"
out_test_pascal_dir = "data_2006/test_patches/pascal"

os.makedirs(out_test_images_dir, exist_ok=True)
os.makedirs(out_test_pascal_dir, exist_ok=True)

# Redirect output dirs for process_image
out_images_dir = out_test_images_dir
out_pascal_dir = out_test_pascal_dir

test_pos_before = total_pos_patches
test_neg_before = total_neg_patches

test_image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg"))
for img_path in test_image_paths:
    label_path = os.path.join(test_pascal_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
    process_image(img_path, label_path=label_path, is_negative_image=False)

test_pos = total_pos_patches - test_pos_before
test_neg = total_neg_patches - test_neg_before
print(f"Test patches: {test_pos} positive, {test_neg} negative")
print(f"Total test patches: {test_pos + test_neg}")
