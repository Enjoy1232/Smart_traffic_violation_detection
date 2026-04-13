# 🚦 Smart Traffic Violation Detection System

## 🌟 Overview
The **Smart Traffic Violation Detection System** is an advanced, AI-powered application designed to monitor traffic and automatically detect various violations in real-time. By leveraging computer vision and deep learning (YOLOv8, DeepSORT, OpenCV), this project enhances road safety and automates the tracking of traffic rule breakers.

## ✨ Features
- **Vehicle Detection & Tracking:** Accurately identifies and tracks cars, motorcycles, bus, trucks, and bicycles across video frames.
- **Traffic Light Signal Detection:** Monitors traffic signals (Red, Yellow, Green) and detects signal violations (e.g., crossing a stop line during a red light).
- **Helmet Detection:** Specifically targets motorcyclists to enforce helmet-wearing regulations.
- **Triple Riding Detection:** Identifies when more than the allowed number of passengers are riding on a single motorcycle.
- **Emergency Vehicle Detection:** Recognizes ambulances, fire engines, and police cars to prevent falsely penalizing them during emergency scenarios.
- **Database Logging:** Asynchronously records all identified violations, along with the timestamps, confidence scores, and vehicle track IDs directly into a MySQL database.

## 🛠️ Technology Stack
- **Detection & Tracking:** [YOLOv8](https://github.com/ultralytics/ultralytics), [DeepSORT](https://github.com/levan92/deep_sort_realtime)
- **Computer Vision:** [OpenCV](https://opencv.org/)
- **Programming Language:** Python 3.9+
- **Database:** MySQL
- **Deep Learning Framework:** PyTorch

## 📁 Project Structure
```text
traffic_violation_system/
├── config.py                 # Core configuration settings (thresholds, DB credentials, tracking vars)
├── db_setup.py               # Script to initialize the MySQL database schema
├── detector.py               # Integration of YOLO models for vehicles, helmets, emergencies
├── main.py                   # Main execution pipeline (video reading, tracking, UI drawing)
├── requirements.txt          # Python dependencies
├── signal_detection.py       # Logic for detecting traffic lights and their state
├── tracker.py                # DeepSORT initialization and setup
├── utils.py                  # Drawing bounding boxes, database inserts, logging
├── violation_logic.py        # Core logic resolving intersections and detecting violations
├── traffic-violation-analysis.pbix # Power BI Dashboard for visualizing recorded violations
├── *.pt                      # Pretrained YOLOv8 PyTorch model files
└── videos/                   # Directory containing sample video footage for inference
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed:
- Python 3.9 or higher
- MySQL Server (Ensure it is running and accessible)

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/Traffic_Violation_System.git
cd Traffic_Violation_System
pip install -r requirements.txt
```

### 3. Database Configuration
1. Open MySQL and ensure your MySQL server is running.
2. Open `config.py` using your preferred text editor.
3. Update the database credentials to match your local MySQL configuration:
   ```python
   # config.py
   DB_HOST = "localhost"
   DB_USER = "your username"
   DB_PASSWORD = os.environ.get("TRAFFIC_DB_PASSWORD", "your password")
   DB_NAME = "traffic_violations"
   ```
4. Run the database setup script to automatically create the database and required tables:
   ```bash
   python db_setup.py
   ```

### 4. Running the System
Place any sample videos you wish to analyze into the `videos/` folder, update the `MODE` or paths in `config.py` if necessary, then run the main application:

```bash
python main.py
```

## 📊 Analytics Dashboard
This project includes a pre-built Power BI dashboard (`traffic-violation-analysis.pbix`) that connects directly to the MySQL database to summarize the logged data.
1. Open the `.pbix` file using [Power BI Desktop](https://powerbi.microsoft.com/desktop/).
2. Click **Refresh** to load the latest logged data from your local MySQL instance.
3. Explore key insights such as peak violation times, frequent offender types, and violation distributions.

## ⚙️ How It Works
1. **Video Ingestion:** `main.py` processes the video frame-by-frame.
2. **Object Detection:** Frames are passed through the primary YOLO instance (`detector.py`) to detect moving targets. Secondary models detect specific elements like traffic lights or motorcycle helmets.
3. **Tracking:** Detected bounding boxes are continuously tracked by the DeepSORT algorithm (`tracker.py`) to maintain individual identities over sequences of frames.
4. **Analysis & Logging:** `violation_logic.py` resolves specific scenarios (e.g., crossing a specified virtual line while the traffic light is red) and flags anomalies.
5. **Storage:** Anomalous entries are sent to MySQL asynchronously so inference time is not bottlenecked (`utils.py`).

## 📜 License
This project is designed for educational and exhibition purposes. Feel free to fork and adapt as needed.

---
**Prepared with ❤️ for a safer tomorrow.**
