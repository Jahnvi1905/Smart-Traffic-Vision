# Smart Traffic Vision 🚦

A Computer Vision college project designed for vehicle detection, multi-object tracking, vehicle counting, violation detection, number plate recognition (OCR), evidence logging, and Streamlit dashboard analytics.

## Project Structure

```text
CV_Project_Traffic/
├── app.py                 # Streamlit dashboard interface
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── detection/             # Vehicle detection module
├── tracking/              # Object tracking module
├── counting/              # Vehicle counting module
├── violations/            # Traffic violation detection module
├── ocr/                   # Number plate OCR module
├── evidence/              # Evidence snapshot generation module
├── database/              # Database management module
├── utils/                 # Utility functions & helpers
└── data/                  # Input/Output data folders
    ├── input/             # Source images/videos
    ├── output/            # Processed media outputs
    └── violations/        # Violation snapshots & evidence
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```
