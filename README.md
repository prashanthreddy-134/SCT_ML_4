# SCT_ML_4: Hand Gesture Recognition (Convolutional Neural Network)

Task 4 of the SkillCraft Technology Machine Learning internship.

**Goal:** recognise hand gestures from images, to support gesture-based human-computer interaction.

**Dataset:** LeapGestRecog (Kaggle, `gti-upv/leapgestrecog`). It contains 10 people (`00` to `09`), each performing 10 gestures, about 20,000 images in total.

**Validation design:** person `09` is held out as the validation set, and the other nine people are used for training. Testing on a person the model has never seen gives a more honest accuracy estimate than a random split, which would put images from the same person in both sets.

## Method
1. Merge the nine training subjects into one folder per gesture.
2. Resize images to 128×128 and augment the training data (small rotations, zoom and shifts).
3. Train a CNN with three Conv–BatchNorm–MaxPool blocks, global average pooling and a dropout-regularised dense layer.
4. Use early stopping, keeping the best weights.
5. Evaluate per-gesture precision, recall and F1 on subject 09.

## Results
Validation accuracy: [value]

| Gesture | Precision | Recall | F1 |
|---|---|---|---|
| [gesture 1] | [value] | [value] | [value] |
| ... | ... | ... | ... |

Full report and confusion matrix: `outputs/metrics.txt`

## How to run
1. Open `SCT_ML_4.ipynb` in Google Colab.
2. Turn on the GPU: Runtime → Change runtime type → T4 GPU.
3. Add your Kaggle username and key when the first cell asks for them.
4. Run all cells (Runtime → Run all). Training takes about 10 to 20 minutes.

## Files
- `SCT_ML_4.ipynb`: Colab notebook with the full run and outputs
- `hand_gesture_cnn.py`: the same training code as a script
- `predict_gesture.py`: classify a single image using the saved model
- `webcam_demo.py`: real-time recognition from a webcam. Not run in Colab, since Colab has no webcam access. It needs a local computer with a camera.
- `outputs/`: trained model, class names, metrics and plots
