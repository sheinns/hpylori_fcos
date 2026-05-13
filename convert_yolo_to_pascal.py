import os
import glob
from PIL import Image

def convert_directory(images_dir, labels_dir, output_dir):
    """Convert YOLO annotations to Pascal VOC for all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    converted_count = 0

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
        output_path = os.path.join(output_dir, img_name.replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            try:
                image = Image.open(img_path)
                w, h = image.size
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue

            with open(label_path, "r") as f:
                lines = f.readlines()

            pascal_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO: class, x_c, y_c, bw, bh (normalized)
                    c, x_c, y_c, bw, bh = map(float, parts)

                    # Convert to absolute
                    x_center = x_c * w
                    y_center = y_c * h
                    box_width = bw * w
                    box_height = bh * h

                    xmin = x_center - (box_width / 2.0)
                    ymin = y_center - (box_height / 2.0)
                    xmax = x_center + (box_width / 2.0)
                    ymax = y_center + (box_height / 2.0)

                    # Format: class xmin ymin xmax ymax
                    pascal_lines.append(f"{int(c)} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")

            with open(output_path, "w") as out_f:
                out_f.writelines(pascal_lines)
            converted_count += 1

    print(f"Converted {converted_count} files in {output_dir}")
    return converted_count

# 1. Convert training set
total = convert_directory(
    "data_2006/dataset/positive/images",
    "data_2006/dataset/positive/labels",
    "data_2006/dataset/positive/pascal"
)

# 2. Convert test set
total += convert_directory(
    "data_2006/dataset-test/images",
    "data_2006/dataset-test/labels",
    "data_2006/dataset-test/pascal"
)

print(f"\nTotal converted: {total} files")
