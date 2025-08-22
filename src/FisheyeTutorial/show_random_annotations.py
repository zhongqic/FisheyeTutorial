import os
import cv2
import yaml
import random
import matplotlib.pyplot as plt

def load_yaml_data(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def get_image_label_pairs(images_dir):
    image_extensions = ('.jpg', '.jpeg','.png', '.PNG')
    image_label_pairs = []

    i = 1
    for root, _, files in os.walk(images_dir):
        for file in files:

            # only sample 9 images
            if i == 10: break
            
            if file.lower().endswith(image_extensions):
                img_path = os.path.normpath(os.path.join(root, file))

                # Build corresponding label path
                if 'images' in img_path:
                    label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
                    
                    if os.path.exists(label_path):
                        image_label_pairs.append((img_path, label_path))
                        i += 1
                    else:
                        continue
                        #print(f"[!] Missing label for: {img_path}")
                else:
                    print(f"[!] Skipping (unexpected path): {img_path}")

    print(f"[✓] Loading {len(image_label_pairs)} image-label pairs.")
    return image_label_pairs

def show_random_annotations(image_label_pairs, class_names=None):
    # Randomly samples and displays images with their YOLO format annotations
  
    selected = random.sample(image_label_pairs, min(9, len(image_label_pairs)))
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    axs = axs.flatten()

    for i, (img_path, lbl_path) in enumerate(selected):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.read().strip().splitlines()

            for line in lines:
                class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
                x1 = int((x_center - box_w / 2) * w)
                y1 = int((y_center - box_h / 2) * h)
                x2 = int((x_center + box_w / 2) * w)
                y2 = int((y_center + box_h / 2) * h)

                color = (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if class_names:
                    label = class_names[int(class_id)]
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(os.path.basename(img_path))

    for j in range(len(selected), 9):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()
