import torch
import sys
import os

# Specify the path to the yolov5 directory
yolov5_path = 'C:/Users/user/Documents/Deep Learning Projects/Formation Detector/yolov5'  # Replace with the actual path to your yolov5 directory

# Add the yolov5 directory to the system path
sys.path.append(os.path.abspath(yolov5_path))

# Import the necessary modules
from models.yolo import Model
from utils.general import check_img_size

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your custom model configuration and weights
model = Model('C:\\Users\\user\\Documents\\Deep Learning Projects\\Formation Detector\\yolov5\\models\\yolov5l.yaml')

# Load the entire state dictionary
state_dict = torch.load('C:\\Users\\user\\Documents\\Deep Learning Projects\\Formation Detector\\yolov5\\runs\\train\\yolov5l_formation\\weights\\best.pt', map_location='cpu')

# Extract the 'model' state dictionary
if 'model' in state_dict:
    model_state_dict = state_dict['model'].state_dict() if hasattr(state_dict['model'], 'state_dict') else state_dict['model']
else:
    model_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

# Filter out unnecessary keys
filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in model.state_dict()}

# Check for missing keys
missing_keys = set(model.state_dict().keys()) - set(filtered_state_dict.keys())
print("Missing keys:", missing_keys)

# Load the filtered state dictionary into the model
model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)  # Ensure the model is moved to the appropriate device (GPU or CPU)
model.eval()  # Set the model to evaluation mode

# Verify the loaded keys
print("Loaded keys:", filtered_state_dict.keys())
