# CNN Image Classification Project

## Project Overview

This project implements a Convolutional Neural Network (CNN) for binary image classification using transfer learning with the InceptionV3 architecture. The model leverages pre-trained weights from ImageNet and adds custom classification layers to perform domain-specific image classification tasks.

## Purpose

- Design and implement a CNN-based image classification system using industry-standard deep learning frameworks
- Utilize transfer learning to leverage state-of-the-art pre-trained models
- Evaluate CNN performance on binary image classification tasks

## Model Architecture

The project uses **InceptionV3** as the base model with the following configuration:

- **Base Model**: InceptionV3 (pre-trained on ImageNet)
- **Feature Extraction Layer**: `mixed7` (output shape: 12×12×768)
- **Input Shape**: 224×224×3 (RGB images)
- **Custom Classification Layers**:
  - Flatten layer
  - Dense layer (512 units, ReLU activation)
  - Dropout (0.2)
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.1)
  - Output layer (1 unit, Sigmoid activation)

**Total Parameters**: 65,730,465 (250.74 MB)
- Trainable: 56,755,201 (216.50 MB)
- Non-trainable: 8,975,264 (34.24 MB)

## Requirements

### Dependencies

```
tensorflow>=2.x
keras>=2.x
numpy
```

### Python Version

- Python 3.7 or higher

## Installation

1. **Clone or download the project**

2. **Install required packages**:

```bash
pip install tensorflow numpy
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

## Project Structure

```
project/
│
├── Training Model.ipynb    # Main training notebook
├── model.h5               # Saved trained model
└── README.md             # This file
```

## How to Run the Project

### Step 1: Prepare Your Environment

Ensure all dependencies are installed:

```bash
pip install tensorflow keras numpy
```

### Step 2: Open the Jupyter Notebook

```bash
jupyter notebook "Training Model.ipynb"
```

### Step 3: Run the Cells

Execute the notebook cells in order:

1. **Import Libraries**: Load TensorFlow, Keras, and other required modules
2. **Load Pre-trained Model**: Initialize InceptionV3 with ImageNet weights
3. **Configure Model**: Freeze base layers and add custom classification layers
4. **Compile Model**: Set up optimizer, loss function, and metrics
5. **View Model Summary**: Inspect the architecture
6. **Save Model**: Export the trained model to `model.h5`

### Step 4: Train the Model (Additional Setup Required)

To train the model, you'll need to add data loading and training code:

```python
# Example: Load your training data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'path/to/train/data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
```

### Step 5: Use the Trained Model

Load and use the saved model for predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('model.h5')

# Load and preprocess an image
img = image.load_img('path/to/image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
prediction = model.predict(img_array)
print(f"Prediction: {prediction[0][0]}")
```

## Model Configuration

### Optimizer
- **Type**: RMSprop
- **Learning Rate**: 0.0001

### Loss Function
- **Binary Crossentropy**: Suitable for binary classification tasks

### Metrics
- **Accuracy**: Primary evaluation metric

## Data Requirements

The model expects:
- **Input Images**: RGB format (3 channels)
- **Image Size**: 224×224 pixels
- **Normalization**: Pixel values should be scaled to [0, 1] range
- **Classification Type**: Binary (two classes)

## Customization

### Changing the Number of Classes

For multi-class classification, modify the output layer:

```python
# Change from binary to multi-class
x = layers.Dense(num_classes, activation='softmax')(x)

# Update the loss function
model.compile(
    optimizer=RMSprop(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Adjusting Model Complexity

Modify the dense layers to suit your needs:

```python
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)  # Increase units
x = layers.Dropout(0.3)(x)                     # Adjust dropout
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)
```

## Performance Tips

1. **Data Augmentation**: Use ImageDataGenerator for better generalization
2. **Learning Rate Scheduling**: Implement callbacks for adaptive learning rates
3. **Early Stopping**: Prevent overfitting with early stopping callbacks
4. **Fine-tuning**: After initial training, unfreeze some InceptionV3 layers for fine-tuning

## Troubleshooting

### Common Issues

**Issue**: TensorFlow warnings about oneDNN optimization
```python
# Add at the beginning of your notebook
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```

**Issue**: Out of memory errors
- Reduce batch size during training
- Use smaller dense layer sizes
- Process images in smaller batches

**Issue**: Poor model performance
- Increase training epochs
- Add data augmentation
- Consider unfreezing more layers for fine-tuning

## License

[Specify your license here]

## Contributors

[Add contributor names]

## Acknowledgments

- InceptionV3 architecture by Google Research
- TensorFlow and Keras development teams
