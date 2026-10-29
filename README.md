# Movie Scene Classification using Deep Learning

Classifies movie frames / scenes (for example forest, street, sea, mountain, buildings)
using transfer learning with **MobileNetV2** in TensorFlow/Keras. It can also scan a
whole movie clip and produce a scene-by-scene timeline.

## Pipeline
Frame -> augmentation -> MobileNetV2 (ImageNet) -> Global Average Pooling -> Dropout -> Softmax

Training happens in two stages: (1) train only the new head, (2) fine-tune the last 30 layers
of MobileNetV2 with a very small learning rate.

Scene classification is **single-label** (one scene per frame), so the model uses
softmax with sparse categorical cross-entropy.

## Dataset
Use any scene dataset arranged in folders. Recommended starting point:
**Intel Image Classification** (Kaggle) with 6 scene classes
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Arrange it like this (rename `seg_train` -> `train`, `seg_test` -> `test`):
```
data/
├── train/
│   ├── buildings/  forest/  glacier/  mountain/  sea/  street/
└── test/
    ├── buildings/  forest/  glacier/  mountain/  sea/  street/
```
To use movie-specific scenes, save frames from clips into your own class folders
(for example `indoor_room`, `car_chase`, `city_night`) and train the same way.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train
```bash
python src/train.py --data data --epochs 8 --fine_tune_epochs 5
```
Outputs in `models/`: `scene_model.keras`, `classes.json`, `confusion_matrix.png`, `accuracy_curve.png`.

## Predict on one image
```bash
python src/predict.py --image path/to/frame.jpg
```

## Predict on a movie clip
```bash
python src/video_predict.py --video videos/clip.mp4 --every 2
```
Prints the scene every 2 seconds and saves `models/scene_timeline.csv`.

## Project structure
```
movie-scene-classification/
├── data/                # train/ and test/ image folders (git-ignored)
├── models/              # saved model, labels, plots
├── videos/              # your test clips (git-ignored)
├── src/
│   ├── data.py          # dataset loading
│   ├── model.py         # MobileNetV2 transfer-learning model
│   ├── train.py         # two-stage training + evaluation
│   ├── predict.py       # single image inference
│   └── video_predict.py # scene timeline for a video
├── requirements.txt
└── README.md
```

## Ideas to improve
- Try EfficientNet or ResNet50 as the backbone
- Add Grad-CAM to visualise what the model looks at
- Build a Streamlit app for uploading frames or clips
- Use Places365 for a wider range of scene categories
