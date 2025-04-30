# RV-Final
Indoor Object Recognition Project
Overview
This project focuses on indoor object recognition using the YOLOv11s model to classify 6 distinct classes: box, broken box, turned on light, turned off light, opened shelf, and closed shelf. The project leverages a dataset of 367 images, annotated with Label Studio, and is implemented using Anaconda, Google Colab, and Python.


Dataset
Size: 367 images
Classes: 6 (box, broken box, turned on light, turned off light, opened shelf, closed shelf)
Annotation Tool: Label Studio was used to label and annotate the images, generating bounding box coordinates and class labels in YOLO format.
Tools and Technologies
Label Studio: For annotating the dataset.
Anaconda: For managing the Python environment and dependencies.
Google Colab: For training the YOLOv11s model with GPU support.
YOLOv11s: A lightweight version of the YOLOv11 model, used for object detection.
Python: Primary programming language for preprocessing, training, and inference.
Project Workflow
1. Data Preparation
Annotation: Images were annotated using Label Studio to create bounding boxes and labels for the 6 classes. The annotations were exported in YOLO format (.txt files with normalized coordinates).
Dataset Split: The 367 images were split into training (80%), validation (10%), and test (10%) sets.




Code Explanation
Label Studio: Used to manually annotate images. It generates .txt files in YOLO format, where each line represents a bounding box as: class_id center_x center_y width height (normalized coordinates).
YOLOv11s Training:
model.train(): This function trains the YOLOv11s model on the dataset. The data parameter points to the dataset.yaml file, epochs controls the number of training iterations, imgsz sets the input image size, and batch defines the batch size.
The model learns to predict bounding boxes and class probabilities for the 6 classes.
Inference:
model.predict(): Takes an image as input, runs the trained model, and outputs bounding boxes with class labels and confidence scores.
results.show(): Visualizes the predictions by drawing bounding boxes on the image.
Google Colab: Provides GPU acceleration, making training faster. The dataset was mounted from Google

Pseudocode
1. Check NVIDIA GPU status
   - Show which GPU is available (optional for Colab environment)

2. Unzip dataset into a custom folder
   - Unzip images and annotations from a .zip file into a new folder named "custom_data"

3. Download utility script for splitting data
   - Get a Python script for splitting images into training and validation sets

4. Run the train/validation split script
   - Total: 367 images
   - Move 330 to train, 37 to validation

5. Install the YOLOv8 framework (Ultralytics)
   - Required for training and inference

6. Define a function to create a YOLO config file data.yaml
   a. Read class names from classes.txt
   b. Create a dictionary:
      - Dataset path
      - Train and validation folders
      - Number of classes
      - Class names
   c. Save it in YAML format to data.yaml

7. Call the function with the correct paths
   - Input: classes.txt file path
   - Output: data.yaml config for YOLO

8. Display contents of data.yaml
   - Example:
     path: /content/data  
     train: train/images  
     val: validation/images  
     nc: 6  
     names: [Box, Closed shelf, ...]

9. Train YOLOv8 model with specified parameters
   - Data: path to data.yaml
   - Model: yolo11s.pt (pretrained)
   - Epochs: 60
   - Image size: 640x640

10. Run object detection prediction on validation images
    - Load the best model from training
    - Save prediction results to a folder

11. Display the first 10 prediction images
    - Use IPython to show predictions in output

12. Save trained model for future use
    a. Create a new folder my_model
    b. Copy trained weights (best.pt) into it
    c. Also copy the training logs

13. Compress trained model and logs into a zip file
    - Creates my_model.zip with model and training outputs

14. Provide download link for the zip file
    - Trigger file download to user's computer from Colab
