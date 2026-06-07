# Document Classification using Convolutional Neural Network (CNN)

A deep learning project that automatically classifies document images into 3 categories using a custom CNN built with PyTorch. The model is deployed as an interactive web application using Streamlit.

---

## What This Project Does

This application accepts a document image as input and predicts what type of document it is. The model has been trained on real document images and can classify them into the following categories:

- **Driving License** — Official government-issued driving identification document
- **Social Security** — Social security card or related document
- **Others** — Any other document that does not fall into the above two categories

The web app shows the predicted class along with a confidence score and probability breakdown for all 3 classes.

---

## Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13 | Main programming language used for all development |
| PyTorch | 2.12.0 | Deep learning framework used to build and train the CNN model |
| Torchvision | 0.27.0 | Used for image transformations and augmentation during training |
| Streamlit | 1.51.0 | Used to build and deploy the interactive web application |
| OpenCV | 4.13.0 | Used for reading and preprocessing document images |
| NumPy | 2.3.5 | Used for numerical operations and array handling |
| Pillow | 12.0.0 | Used for opening and handling image files in the app |
| Scikit-learn | 1.8.0 | Used for generating confusion matrix and classification report |
| Matplotlib | 3.10.0 | Used for plotting training results and confusion matrix |
| Pandas | 2.3.3 | Used for data handling and analysis |

---

## Project Structure

```
Dacument_classification_project/
│
├── app.py                  --> Streamlit web application (main file to run)
├── requirements.txt        --> List of all required Python libraries with versions
├── README.md               --> Full project description and documentation
│
├── output/
│   └── model.pt            --> Trained CNN model saved after best validation accuracy
│
└── Data/
    ├── Training_data/      --> Contains training images organized in class folders
    |     ├── Driving_license/
    |     ├── Others/
    |     └── Social_security/
    └── Testing_Data/       --> Contains testing images organized in class folders
          ├── Driving_license/
          ├── Others/
          └── Social_security/
```

---

## Model Architecture

The CNN model is built from scratch using PyTorch. It consists of the following layers:

| Layer | Details |
|-------|---------|
| Conv Layer 1 | 3 input channels, 32 filters, kernel size 3x3, padding 1 |
| Batch Normalization | Applied after each conv layer to stabilize training |
| MaxPooling | 2x2 pooling to reduce spatial dimensions |
| Conv Layer 2 | 32 input channels, 64 filters, kernel size 3x3, padding 1 |
| Conv Layer 3 | 64 input channels, 128 filters, kernel size 3x3, padding 1 |
| AdaptiveAvgPool | Reduces output to 1x1 regardless of input size |
| Fully Connected 1 | 128 input features to 64 output features |
| Dropout | 0.3 dropout rate to prevent overfitting |
| Fully Connected 2 | 64 input features to 3 output classes |

---

## Training Details

| Detail | Value |
|--------|-------|
| Input Image Size | 200 x 200 pixels |
| Number of Classes | 3 |
| Optimizer | Adam with learning rate 0.0003 |
| Loss Function | Cross Entropy Loss |
| Batch Size | 16 |
| Max Epochs | 20 |
| Early Stopping | Yes, patience of 7 epochs |
| LR Scheduler | ReduceLROnPlateau with patience 3 |

---

## Data Augmentation

Since the dataset was small, data augmentation was applied during training to increase the effective dataset size by 5 times. The following techniques were used:

- Random Horizontal Flip (50% probability)
- Random Vertical Flip (20% probability)
- Random Rotation up to 15 degrees
- Color Jitter for brightness, contrast and saturation
- Random Grayscale (10% probability)

Augmentation was only applied to training data. Testing data was never augmented to ensure fair evaluation.

---

## How to Run Locally

**Step 1 - Clone the repository**
```bash
git clone https://github.com/your-username/Dacument_classification_project.git
cd Dacument_classification_project
```

**Step 2 - Install required libraries**
```bash
pip install -r requirements.txt
```

**Step 3 - Run the Streamlit app**
```bash
streamlit run app.py
```

**Step 4 - Open in browser**
```
http://localhost:8501
```

---

## How to Use the App

1. Open the app in your browser
2. Click on "Upload Document Image" button
3. Select any document image from your computer (JPG or PNG format)
4. The app will instantly show the predicted document type
5. You can also see the confidence score and probability for each class

---

## Author

This project was developed as part of a deep learning study on document classification using Convolutional Neural Networks.

---

## License

This project is open source and is available for educational and research purposes only.
