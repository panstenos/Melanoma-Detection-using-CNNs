# Melanoma-Detection-using-CNNs
Training CNNs to classify malignant from benign melanomas

<figure style="float:left; margin-right: 10px;">
    <img src="https://github.com/panstenos/Melanoma-Detection-using-CNNs/assets/112823396/785079d5-1bed-4d62-982b-5fb8fa9f2c08">
    <figcaption>Benign Melanoma</figcaption>
</figure>

<figure style="float:left; margin-right: 10px;">
    <img src="https://github.com/panstenos/Melanoma-Detection-using-CNNs/assets/112823396/64ab050e-4f08-4fbf-bb72-4e8ab23faa25">
    <figcaption>Malignant Melanoma</figcaption>
</figure>



## Context
Melanoma, a lethal form of skin cancer, poses a significant public health challenge worldwide. Early detection and accurate diagnosis are critical for effective treatment and improved patient outcomes. This dataset serves as a valuable resource for researchers and healthcare practitioners striving to advance the field of dermatology and oncology.

Harnessing cutting-edge technology, this dataset offers a diverse collection of dermatoscopic images capturing both benign and malignant skin lesions. Each image is meticulously annotated by expert dermatologists, providing invaluable ground truth for algorithmic analysis.

By leveraging this dataset, researchers can develop and evaluate machine learning algorithms capable of automatically detecting and classifying melanocytic lesions with high precision and recall. Such advancements in computer-aided diagnosis hold the potential to revolutionize melanoma screening programs, enabling earlier detection, personalized treatment plans, and ultimately, saving lives.

Moreover, the availability of this dataset fosters collaboration and innovation within the scientific community, driving forward the development of robust and interpretable models that enhance clinical decision-making and patient care.

Together, through the utilization of this dataset and the collaborative efforts of researchers and practitioners worldwide, we can accelerate progress in the fight against melanoma, ultimately striving towards a future where no life is lost to this devastating disease.

## Sources and Inspiration
This dataset draws inspiration from the critical need for advanced diagnostic tools in dermatology. The images are compiled from diverse sources and showcase the intricate features that challenge traditional diagnostic methods. By sharing this dataset on Kaggle, we invite the global data science community to collaborate, innovate, and contribute towards developing reliable models for melanoma classification.

## About the Dataset
This dataset, consists of 13,900 meticulously curated images of a uniform size of (224, 224, 3). This is a valuable resource for advancing the field of dermatology and computer-aided diagnostics. The data are split as follows:

|                    | Train Images | Test Images |
|--------------------|--------------|-------------|
| Benign             | 6289         | 1000        |
| Malignant          | 5590         | 1000        |

To check out the data set [click here!](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data)

## Data Collection 
Since I am working at a Colab notebook I uploaded the zip folder of the data on my google drive and then unziped the folder using the following commands:
```python
# mount drive to colab
from google.colab import drive
drive.mount('/content/drive')

# unzip the data
%cd '/content/drive/MyDrive/Colab Notebooks/Melanoma Cancer Detection'
!unzip data.zip
```

## Modelling
I explored 3 models. A custom CNN, AlexNet and InceptionV3. All models had an input shape of (224, 224, 3). I also defined a callback to terminate training when the training accuracy exceeded 99.0%.
```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99.0% accuracy -> Terminating Training")
      self.model.stop_training = True
```
you call this function when you fit the model like so:
```python
history = model.fit(..., callbacks=[callbacks])
```

## Model Performance Comparison

For all training, RMSprop with a learning rate of 0.001 was used.

### Preprocessing A:

I initially performed multiple augmentations to the training images:
```python
train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=40,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.05,
                                   zoom_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(directory=train_dir, batch_size=20, class_mode='binary', target_size=(224,224))
```
For the test dataset:
```python
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(directory=test_dir, batch_size=20, class_mode='binary', target_size=(224,224))
```

Fitting the models:
```python
history = model.fit(train_generator, steps_per_epoch=25, epochs=100, verbose=1, validation_data=test_generator, validation_steps=5, callbacks=[callbacks])
```

| Model          | Train Accuracy | Test Accuracy | Time per Epoch |
|----------------|----------------|---------------|----------------|
| Custom CNN     | 0.8780         | 0.8700        |   9s           |
| AlexNet        | 0.8640         | 0.8600        |   9s           |
| InceptionV3    | 0.8880         | 0.9300        |  11s           |

### Preprocessing B:

For the second preprocessing, I did not perform any additional augmentations to the training dataset:
```python
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(directory=train_dir, batch_size=50, class_mode='binary', target_size=(224,224))
```
For the test dataset:
```python
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(directory=test_dir, batch_size=50, class_mode='binary', target_size=(224,224))
```

Fitting the models:
```python
history = model.fit(train_generator, steps_per_epoch=50, epochs=50, verbose=1, validation_data=test_generator, validation_steps=10, callbacks=[callbacks])
```

| Model          | Train Accuracy | Test Accuracy | Time per Epoch |
|----------------|----------------|---------------|----------------|
| Custom CNN     | 0.8844         | 0.8920        |   14s          |
| AlexNet        | 0.8658         | 0.8560        |   14s          |
| InceptionV3    | 0.9920         | 0.9260        |   15s          |
