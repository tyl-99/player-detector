import cv2
import numpy as np
import torch
import cupy as cp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\user\\Documents\\Deep Learning Projects\\Formation Detector\\yolov5\\runs\\train\\formationmodel5\\weights\\best.pt')
model.to(device)
model.eval()

colors = []
main_color = []

def get_two_major_colors(rgb_tuples):
    pixels = np.float32(rgb_tuples)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    k = 2
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    major_colors = palette.astype(int)
    
    return [tuple(color) for color in major_colors]

def get_dominant_color(image):
    global main_color
    def find_nearest_color(target_color, color_list):
        distances = [np.linalg.norm(np.array(target_color) - np.array(color)) for color in color_list]
        nearest_index = np.argmin(distances)
        return color_list[nearest_index]

    if image.size == 0:
        return (0, 0, 0)

    height, width, _ = image.shape
    
    center_y, center_x = height // 2, width // 2
    
    center_region = image[0:center_y, center_x-2:center_x+2]
    
    center_region = cv2.resize(center_region, (50, 50), interpolation=cv2.INTER_AREA)
    pixels = np.float32(center_region.reshape(-1, 3))
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = palette[0].astype(int)
    dominant_color_tuple = tuple(dominant_color)
    
    #return dominant_color_tuple
    if not main_color:
        return dominant_color_tuple
    else:
        return find_nearest_color(dominant_color_tuple, main_color)

def non_max_suppression(detections, iou_threshold=0.7):
    if len(detections) == 0:
        return []
    
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep_detections = []

    while detections:
        best_detection = detections.pop(0)
        keep_detections.append(best_detection)
        detections = [d for d in detections if iou(best_detection, d) < iou_threshold]

    return keep_detections

def iou(det1, det2):
    x1_1, y1_1, x2_1, y2_1, _, _ = det1
    x1_2, y1_2, x2_2, y2_2, _, _ = det2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area

video_path = "C:\\Users\\user\\Documents\\Deep Learning Projects\\Formation Detector\\newtest.mp4"
cap = cv2.VideoCapture(video_path)
frames = []
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

init_color = False

def inf():
    global init_color,main_color,colors
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        oheight, owidth = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640, int(640 * 64 / 64)))

        image_normalized = frame_resized / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            results = model(image_tensor)

        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)

        detections = results[0]
        threshold = 0.45
        mask = detections[:, 4] > threshold
        filtered_detections = detections[mask]

        x_center = filtered_detections[:, 0].cpu().int()
        y_center = filtered_detections[:, 1].cpu().int()
        width = filtered_detections[:, 2].cpu().int()
        height = filtered_detections[:, 3].cpu().int()
        conf = filtered_detections[:, 4]
        class_conf = filtered_detections[:, 5]

        x1 = (x_center - width // 2).numpy()
        y1 = (y_center - height // 2).numpy()
        x2 = (x_center + width // 2).numpy()
        y2 = (y_center + height // 2).numpy()

        detections = list(zip(x1, y1, x2, y2, conf.cpu().numpy(), class_conf.cpu().numpy()))

        filtered_detections = non_max_suppression(detections)

        

        for detection in filtered_detections:
            x1, y1, x2, y2, conf, class_conf = detection
            player_id = (x1, y1, x2, y2)

            w = x2 - x1
            h = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            y_start, y_end = min(y1, y2), max(y1, y2)
            x_start, x_end = min(x1, x2), max(x1, x2)

            player_roi = image_np[y_start:y_end, x_start:x_end]

            if player_roi.size == 0:
                continue  


            dominant_color = get_dominant_color(player_roi)

            if(not init_color):
                colors.append(dominant_color)

            oval_center = (center_x, y2)
            ellipse_mask = np.zeros_like(image_np, dtype=np.uint8)
            cv2.ellipse(ellipse_mask, oval_center, (12, 9), 0, 0, 360, (255, 255, 255), 2)

            player_mask = np.zeros_like(image_np, dtype=np.uint8)
            cv2.rectangle(player_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

            final_mask = cv2.bitwise_and(ellipse_mask, cv2.bitwise_not(player_mask))

            # Apply dominant color to the ellipse region
            ellipse_color = (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))
            mask_indices = np.where(final_mask[:, :, 0] == 255)
            image_np[mask_indices] = ellipse_color

        frames.append(cv2.resize(image_np, (owidth, oheight), interpolation=cv2.INTER_LINEAR))

        if(not init_color):
            main_color = get_two_major_colors(colors)
            init_color = True

    cap.release()

inf()

output_path = "C:\\Users\\user\\Documents\\Deep Learning Projects\\Formation Detector\\output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

for frame in frames:
    out.write(frame)

out.release()
cv2.destroyAllWindows()
