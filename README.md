# 👁️ Eye Diseases Classification Using InceptionV3

This project aims to classify various eye diseases using transfer learning with the **InceptionV3** model. The dataset used is sourced from Kaggle, and the model achieves high accuracy in detecting diseases from retina images.

---

## 📂 Dataset

You can download the dataset from Kaggle here:  
🔗 [Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data)

---

## 🛠️ Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/your-username/eye-disease-classification.git
cd eye-disease-classification
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install manually:

```bash
pip install tensorflow numpy matplotlib scikit-learn opencv-python
```

---

## 📁 Project Structure

```
.
├── eye-diseases-classification-inceptionv3-95.ipynb  # Jupyter Notebook for training and testing
├── model/                                            # (Optional) Saved trained model
├── images/                                           # Sample predictions
└── README.md                                         # Project documentation
```

---

## 🧠 Model: InceptionV3 (Transfer Learning)

This project uses the **InceptionV3** model pretrained on ImageNet. Only the final layers are retrained to adapt to the eye disease classification task.

### Key steps:
- Image preprocessing and augmentation
- Transfer learning using InceptionV3
- Training on retina images with class labels

---

## 🧪 How to Train the Model

Inside the Jupyter notebook:

```python
# Load the pretrained InceptionV3 base model
from tensorflow.keras.applications import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

Freeze base layers:

```python
for layer in base_model.layers:
    layer.trainable = False
```

Add custom classifier layers:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
```

Compile and train:

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)
```

---

## 📊 Results

- Final Accuracy: **~95%**
- Model: **InceptionV3**
- Classification: Multi-class (e.g., Cataract, Glaucoma, Normal, etc.)

---

## 📈 Sample Prediction

You can test the model with:

```python
img = load_img('path_to_image.jpg', target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
```

---

## 🧾 License

This project is licensed under the MIT License.
