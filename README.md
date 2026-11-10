# SCT_ML_4 – Hand Gesture Recognition (CNN)

**Task:** Develop a hand gesture recognition model that identifies and classifies different hand gestures from image or video data, enabling intuitive human-computer interaction.
**Dataset:** Hand Gesture Dataset (link in the internship brief).

## Setup
```bash
pip install -r requirements.txt
```
Unzip the dataset into `data/` with one sub-folder per gesture class (for example `data/thumbs_up/`, `data/peace/`). If the dataset is organised differently, reorganise it into this layout first.

## Run
```bash
python hand_gesture_cnn.py --data data --epochs 15     # train and evaluate
python predict_gesture.py path/to/gesture.jpg           # classify one image
python webcam_demo.py                                   # real-time demo from the webcam
```

## Method
1. Load images with an 80/20 train/validation split and resize to 128×128.
2. Augment training images (small rotations, zoom and shifts).
3. Train a CNN with three Conv–BatchNorm–MaxPool blocks, global average pooling and a dropout-regularised dense head.
4. Use early stopping, then report per-class precision, recall and F1 with a confusion matrix.
5. Use the trained model for single-image prediction and live webcam recognition.

## Outputs (`outputs/`)
- `gesture_model.keras`: trained model
- `class_names.json`: class labels in model order
- `training_curves.png`, `confusion_matrix.png`, `metrics.txt`

## Results
Validation accuracy: ____
