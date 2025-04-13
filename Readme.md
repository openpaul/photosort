# PhotoSort: Image Classifier

This project uses a machine learning model to classify images within a given folder. 
It can then organize these images into subfolders based on the predicted category and (optionally) the date the image was taken.

## Features

* **Image Classification:** Uses a pre-trained CLIP model to generate image embeddings and a classical ML model (Random Forest by default) to predict image categories.
* **Video Support:** Extracts frames from videos and classifies them, using a majority vote for the final video category.
* **Organized Output:** Sorts classified images into folders based on predicted category and date (year/month).
* **Move or Copy:** Can either move or copy the classified images to the output directory.
* **Dry Run:** Allows for a test run without actually moving or copying files.
* **Training:** Enables training a new classification model on a labeled image dataset.

## Requirements

* Python 3.12 or higher
* Dependencies listed in the script (installable via `uv run --script photosort.py`)
* `ffmpeg` installed on your system for video processing.

## Usage

1.  **Clone the repository** (if applicable) or save the provided Python script.
2.  **Install dependencies:** Run the script directly (e.g., `uv run --script photosort.py`). This will handle the installation of the required libraries.
3.  **Training (Optional):**
    ```bash
    python your_script.py train <path_to_labeled_image_folder>
    ```
    Replace `<path_to_labeled_image_folder>` with the path to your folder containing subfolders of labeled images.
4.  **Classification:**
    ```bash
    python your_script.py classify -i <path_to_input_folder> -o <path_to_output_folder>
    ```
    * `-i` or `--input`: Path to the folder containing images and videos to classify.
    * `-o` or `--output`: Path to the folder where classified images will be organized.
    * Other optional arguments like `-m` (move), `-d` (dry-run), `-f` (force overwrite), and `-k` (video frame sampling) are available.

## Notes

* The first level of subfolders within the training data directory will be used as image labels.
* The classification model is saved in the application's data directory.
* Empty subfolders in the input directory can be automatically deleted after classification.

This is a basic overview. For more details and options, run the script with the `--help` flag for each command (e.g., `python photosort.py classify --help`).

### Training data structure

Sort your training data like this. Labels are extracted from folder names.

```txt
training_data/
├── cats/
│   ├── cat_image_01.jpg
│   ├── cat_photo.png
│   └── fluffy_cat.jpeg
└── dogs/
    ├── dog_picture.jpg
    ├── golden_retriever.png
    └── puppy.heic
```