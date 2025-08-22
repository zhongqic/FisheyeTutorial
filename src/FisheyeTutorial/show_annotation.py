def draw_yolo_boxes(image_path, label_path, class_names=None):
    # Displays a particular image with its YOLO format annotations
  
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Read YOLO labels
    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        color = (255, 0, 0)  # Red
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if class_names:
            label = class_names[int(class_id)]
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    plt.imshow(img)
    plt.axis('off')
    plt.title("Image with Bounding Boxes")
    plt.show()
