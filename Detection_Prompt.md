# Task: Prepare a Reproducible COCO 10K Dataset for Object Detection

Create a Python script and Jupyter Notebook to prepare a reproducible 7-class object detection dataset based on COCO 2017.

The dataset will be used to train and compare multiple object detection models in terms of detection accuracy and computational efficiency.

**All students must generate exactly the same dataset, including image selection, train/validation splits, class mappings, and annotations. Follow the requirements below without changing any settings.**

## 1. Existing dataset

The following files have already been downloaded into the `coco/` directory:

- `train2017.zip`
- `val2017.zip`
- `annotations_trainval2017.zip`

Do not download these files again.

Read the ZIP archives directly and extract only the images needed for this experiment.

Use:

- `instances_train2017.json` for training annotations.
- `instances_val2017.json` for validation annotations.

## 2. Dataset configuration

Create a dataset containing exactly **10,000 images**:

- **Training: 9,000 images** from COCO train2017.
- **Validation: 1,000 images** from COCO val2017.

Use exactly these seven classes:

| Class ID | Class Name | Original COCO Category ID |
|---|---|---|
| 0 | person | 1 |
| 1 | car | 3 |
| 2 | bicycle | 2 |
| 3 | dog | 18 |
| 4 | cat | 17 |
| 5 | bus | 6 |
| 6 | truck | 8 |

Remove annotations belonging to other classes.

Keep images even if they contain no annotations for the selected classes.

## 3. Deterministic image selection

Ensure every student selects exactly the same images, regardless of operating system or Python version.

For each split independently:

1. Read all original COCO image IDs.
2. Calculate SHA-256 for each image ID using the string `coco7-v1:{split}:{image_id}`, where `split` is `train` or `val`.
3. Sort images by their hexadecimal SHA-256 values in ascending order, using numeric image ID as a tie-breaker.
4. Select the first **9,000 images** from the sorted train2017 records.
5. Select the first **1,000 images** from the sorted val2017 records.

Do not use random sampling, stratified sampling, or a library-specific splitting function.

Save the selected image IDs in `split_manifest.json`.

## 4. Annotation conversion

Convert the COCO annotations into Ultralytics YOLO Detection format.

For each image:

1. Retain only annotations corresponding to the seven target classes.
2. Exclude annotations with `iscrowd=1`.
3. Remap original COCO category IDs to class IDs 0–6.
4. Clip bounding boxes to valid image boundaries.
5. Skip invalid or zero-area bounding boxes.
6. Convert COCO bounding boxes into normalized YOLO coordinates.
7. Sort annotations by their original numeric annotation IDs.
8. Save coordinates with six decimal places.
9. Exclude boxes that become zero-area after rounding.
10. Create empty label files for images without target-class annotations.

Each annotation must follow:

`class_id x_center y_center width height`

All bounding box coordinates must be normalized to `[0, 1]`.

## 5. Output directory structure

Create:

```text
datasets/
  coco7_10k/
    images/
      train/
      val/
    labels/
      train/
      val/
    coco7.yaml
    split_manifest.json
```

Keep original image filenames.

Do not resize images during dataset preparation.

## 6. Dataset YAML configuration

Generate `coco7.yaml` with:

```yaml
train: images/train
val: images/val

nc: 7

names:
  0: person
  1: car
  2: bicycle
  3: dog
  4: cat
  5: bus
  6: truck
```

Set `path` to the absolute path of the generated dataset directory.

Ensure the dataset is compatible with Ultralytics YOLO training and validation.

## 7. Dataset verification

Automatically verify:

- Exactly **9,000 training images** and **1,000 validation images**.
- No overlap between training and validation images.
- Each image has a corresponding label file.
- All class IDs belong to 0–6.
- All bounding box coordinates are valid.
- All seven classes are represented in both splits.

Print:

- Number of images in each split.
- Number of bounding boxes per class.
- Number of images containing each class.
- Number of images with empty annotations.

Generate deterministic SHA-256 fingerprints for the image-ID lists and annotation contents.

Save the fingerprints in `split_manifest.json`.

If verification fails, report the problem rather than changing the image selection rules.

## 8. Visualization

Visualize 10 images with their bounding boxes and class labels.

Select the visualization samples deterministically using seed 42.

## 9. Evaluation Metrics for Future Model Comparisons

The prepared dataset will be used to compare multiple object detection models in terms of detection accuracy and computational efficiency.

All models must be evaluated using the same validation dataset and evaluation settings.

### Detection Accuracy

**Primary Metric:**

- mAP@50:95

**Secondary Metrics:**

- mAP@50
- Precision
- Recall

Use standard COCO-style IoU thresholds from 0.50 to 0.95 with a step of 0.05.

### Model Efficiency

**Primary Metric:**

- **CPU Utilization (%)**

**Secondary Metrics:**

- Inference Latency (ms/image)
- Number of Parameters (M)
- GFLOPs

#### CPU Utilization Measurement Requirements

CPU utilization must represent the actual CPU resources used by each object detection model during inference.

For fair comparison:

1. Run inference on the CPU only, without GPU acceleration.
2. Use the same physical CPU and hardware environment for all models.
3. Use an input image size of 640×640.
4. Use a batch size of 1.
5. Use the same number of CPU inference threads for every model.
6. Use the same inference backend and preprocessing/postprocessing settings.
7. Perform 30 warm-up iterations before measurement.
8. Run inference on the same fixed set of 200 validation images, selected deterministically by ascending original COCO image ID.
9. Measure CPU usage periodically during inference using `psutil` or an equivalent tool.
10. Report the average and peak CPU utilization (%).
11. Measure the CPU usage of the inference process, including its child processes, rather than relying only on total system CPU usage.
12. Normalize process CPU utilization by the number of logical CPU cores to express utilization on a 0–100% whole-machine scale.
13. Record the CPU model, logical core count, thread count, and inference backend.

**The primary Model Efficiency metric is Average CPU Utilization (%).**

Also report inference latency (ms/image), because lower CPU utilization does not necessarily imply more efficient inference if the model takes longer to process each image.

The final comparison should include all detection accuracy and model efficiency metrics.

Do not train or evaluate any models yet. These metrics define the requirements for future model comparisons.

## 10. Deliverables

Generate:

1. `prepare_coco7.py` — Dataset preparation script.
2. `prepare_coco7.ipynb` — Jupyter Notebook for dataset preparation and verification.
3. `coco7.yaml` — Detection dataset configuration.
4. `split_manifest.json` — Dataset split information and fingerprints.
5. `README.md` — Instructions for reproducing the dataset.

The script and notebook must produce identical results.

The notebook must run from top to bottom without requiring manual modification of the generated annotations.

**The goal is to prepare a standardized COCO 10K dataset that supports fair and reproducible comparisons across different object detection models.**
