# Vehicle Detection and Counting using HOG and Hough Transform

## Overview

This project focuses on detecting and counting four-wheeled vehicles crossing an intersection in video footage using **Histogram of Oriented Gradients (HOG)** for object detection and **Hough Transform** for line detection. 

---

## Dataset

The dataset is located in the `data1` folder and consists of:

- **Videos** (`videos/`): Video files containing vehicles crossing an intersection.
- **Images** (`pictures/`): Training images for positive (vehicles) and negative samples.
- **Ground Truth Counts** (`counts.csv`): The correct number of vehicle crossings for each video.

---

## Project Structure

```
📂 Vehicle-Counting
 ├── 📂 pictures          # Training images (positive & negative samples)
 ├── 📂 videos            # Videos of intersections with vehicle crossings
 ├── 📝 counts.csv        # Ground truth vehicle counts
 ├── 📜 SC23-G3-RA-186-2020.py    # Main Python script for detection and counting
 ├── 📜 README.md             # Project documentation
```

---

## Requirements

To run this project, install the necessary dependencies:

```bash
pip install numpy opencv-python scikit-learn matplotlib pandas
```

---

## Implementation Details

### **1. Vehicle Detection**

- The **HOG descriptor** is used to extract features from training images.
- A **Support Vector Machine (SVM)** classifier is trained to distinguish vehicles from non-vehicles.
- A **sliding window approach** is applied to detect vehicles in video frames.

### **2. Intersection Line Detection**

- **Canny Edge Detection** is applied to extract edges.
- **Hough Line Transform** is used to detect the crossing line in the intersection.

### **3. Counting Vehicles**

- Each detected vehicle is checked for crossing the line.
- The total number of crossings is stored and compared to the ground truth from `counts.csv`.
- The accuracy of the detection is measured using **Mean Absolute Error (MAE)**.

---

## How to Run

1. Place the dataset in the `data1` folder.
2. Run the script:

```bash
python SC23-G3-RA-186-2020.py videos
```

3. The script will process all videos and output the vehicle counts along with the MAE.

---

## Evaluation & Results

- The model's performance is evaluated using **Mean Absolute Error (MAE)**.
- A target **MAE ≤ 3.5** is required for full marks (22 points).
- The results are displayed as:
  ```
  video1.mp4 - Ground Truth: 10 - Predicted: 9
  video2.mp4 - Ground Truth: 15 - Predicted: 13
  Mean Absolute Error: 2.0
  ```

---

## Future Improvements

- Implement **Deep Learning-based** vehicle detection for improved accuracy.
- Enhance image preprocessing to handle different lighting and occlusion conditions.
- Fine-tune SVM parameters and feature extraction methods.

---

## Author

- **Katarina Perović**

---

## License

This project is for academic purposes only as part of \*\*Soft Computing 2023/24 \*\*

