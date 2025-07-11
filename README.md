# Re-identification_in_single_feed Pipeline

An end-to-end Python toolkit for analyzing football (soccer) videos. This pipeline:

* Tracks **players**, **referees**, and the **ball** using YOLOv11 + ByteTrack
* Compensates for **camera motion** via optical flow
* Applies a **perspective transform** to map image points to real-world field coordinates
* Identifies **teams** by clustering jersey colors (K-Means)
* Assigns **ball possession** to the nearest player
* Outputs a fully **annotated video** with colored bounding boxes

---

## 📂 Repository Structure

```text
football_analysis-main/
├── input_videos/               # Place your .mp4 files here
├── output_videos/              # Annotated videos saved here
├── models/                     # Pretrained YOLO weights (e.g. best.pt)
├── stubs/                      # Optional cache for tracking & motion
├── requirements.txt            # Python dependencies
├── main.py                     # Pipeline orchestrator
├── README.md                   # This guide
├── utils/                      # I/O and geometry helpers
├── trackers/                   # Detection + ByteTrack + annotation
├── camera_movement_estimator/  # Optical flow motion estimation
├── view_transformer/           # Homography transformer
├── team_assigner/              # K-Means jersey color clustering
└── player_ball_assigner/       # Nearest-player ball assignment
```

---

## ⚙️ Setup

1. **Clone repository**

   ```bash
   git clone https://github.com/KunalNath04/Re-identification_in_single_feed.git
   cd Re-identification_in_single_feed
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Add your video** to `input_videos/` (e.g. `match.mp4`).
2. **Run the pipeline**:

   ```bash
   python main.py
   ```
3. **Check output** in `output_videos/output_video.avi`.

---

## 🔍 Pipeline Steps

1. **Video Loading**: Reads frames from the `.mp4` file.
2. **Detection & Tracking**: YOLOv11 + ByteTrack assigns consistent IDs.
3. **Camera Motion Compensation**: Removes global camera shifts.
4. **Perspective Transform**: Maps image to field coordinates.
5. **Team Clustering**: K-Means groups players into two teams by jersey color.
6. **Ball Possession**: Assigns the ball to the closest player each frame.
7. **Annotation**: Draws bounding boxes:

   * **Team 1**: Blue
   * **Team 2**: Red
   * **Referees**: Yellow
   * **Ball**: Green (only when detected)
8. **Save Video**: Exports annotated results.

---

## 🔧 Configuration & Tips

* **Model Path**: Change the YOLO weights path in `main.py` if needed.
* **Clustering**: Tune K-Means `n_init` in `team_assigner/team_assigner.py` for color variation.
* **Motion Estimation**: Adjust optical-flow parameters in `camera_movement_estimator`.

---


