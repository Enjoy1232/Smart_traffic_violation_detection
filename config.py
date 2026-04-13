import os

#Performance Mode 
GPU_MODE = False

#Database
DB_HOST = "localhost"
DB_USER = "your username"
DB_PASSWORD = os.environ.get("TRAFFIC_DB_PASSWORD", "your password")
DB_NAME = "traffic_violations"

#Operating Mode

MODE = "video"          
VIDEO_FOLDER   = "videos"

#Model
MODEL_PATH = "yolov8s.pt"         
HELMET_MODEL_PATH = "helmet_model.pt"   
EMERGENCY_MODEL_PATH = "emergency_model.pt"
EMERGENCY_CONF_THRESHOLD = 0.65
EMERGENCY_KEYWORDS = ["AMBULANCE", "FIRE", "POLICE", "108", "101", "100"]
YOLO_CONF_THRESHOLD = 0.35            
YOLO_IOU_THRESHOLD  = 0.45            

#Inference Performance
INFERENCE_INPUT_WIDTH  = 640
INFERENCE_INPUT_HEIGHT = 640

YOLO_IMGSZ = 416

# Device for YOLO inference.
YOLO_DEVICE = "cuda" if GPU_MODE else "cpu"

DETECT_EVERY_N_FRAMES = 1 if GPU_MODE else 1

#COCO class
VEHICLE_CLASS_IDS = {
    2:  "Car",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
    1:  "Bicycle",
}

PERSON_CLASS_ID        = 0
TRAFFIC_LIGHT_CLASS_ID = 9
CELL_PHONE_CLASS_ID    = 67

#DeepSORT
DEEPSORT_MAX_AGE       = 60 if GPU_MODE else 30   
DEEPSORT_N_INIT        = 2 if GPU_MODE else 1    
DEEPSORT_MAX_COSINE_DISTANCE = 0.4
ID_SWITCH_GRACE_PERIOD_SEC   = 1.5  
ID_SWITCH_MAX_DISTANCE       = 70   

#Traffic Signal Detection 
SIGNAL_ROI = (10, 10, 120, 120)

# HSV ranges for signal colours
SIGNAL_HSV_RED_LOWER1 = (  0, 120,  70)
SIGNAL_HSV_RED_UPPER1 = ( 10, 255, 255)
SIGNAL_HSV_RED_LOWER2 = (170, 120,  70)
SIGNAL_HSV_RED_UPPER2 = (180, 255, 255)
SIGNAL_HSV_YELLOW_LOWER = (15, 100, 100)
SIGNAL_HSV_YELLOW_UPPER = (35, 255, 255)
SIGNAL_HSV_GREEN_LOWER = (40, 60, 60)
SIGNAL_HSV_GREEN_UPPER = (90, 255, 255)

SIGNAL_MIN_PIXEL_COUNT = 200           
SIGNAL_BUFFER_SIZE     = 5             

# Confidence threshold 
TRAFFIC_LIGHT_CONF_THRESHOLD = 0.35
TRAFFIC_LIGHT_MIN_CROP_SIZE  = 15
TRAFFIC_LIGHT_CLASSIFIER_CONF= 0.60
TRAFFIC_LIGHT_TRACK_FRAMES   = 5

#Stop Line 
STOP_LINE_Y_FRAC = 0.85


# Helmet / Rider Detection
HELMET_DETECTION_ENABLED = True

TRIPLE_RIDING_MAX_DISTANCE_FRAC: float = 0.25

HELMET_OVERLAP_THRESHOLD = 0.40
HELMET_SKIN_FRACTION_THRESHOLD   = 0.12
HEAD_REGION_FRACTION = 0.50
HELMET_CONF_THRESHOLD = 0.30
HELMET_MIN_CONFIDENCE = 0.30
NO_HELMET_FRAMES_THRESHOLD = 3
TRIPLE_RIDING_PERSON_LIMIT = 2         
RIDER_VERTICAL_SLACK_FRAC = 0.08      

# Display 
WINDOW_NAME        = "Traffic Violation Detection"
DISPLAY_WIDTH      = 1280
DISPLAY_HEIGHT     = 720
BOX_COLOR_OK       = (0, 200, 0)      
BOX_COLOR_VIOLATION= (0,   0, 255)     
BOX_THICKNESS      = 2
FONT_SCALE         = 0.55
FONT_THICKNESS     = 1

# Video
SAVE_OUTPUT_VIDEO  = False
OUTPUT_VIDEO_DIR   = "output_videos"
