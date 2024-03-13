# Melanoma-Detection-using-CNNs
Training CNNs to classify malignant from benign melanomas

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

## Preprocessing 
Since I am working at a Colab notebook I uploaded the zip folder of the data on my google drive and then unziped the folder using the following commands:
```python
# mount drive to colab
from google.colab import drive
drive.mount('/content/drive')

# unzip the data
%cd '/content/drive/MyDrive/Colab Notebooks/Melanoma Cancer Detection'
!unzip data.zip
```

## Image Augmentation
I used the ImageDataGenerator class to load and preprocess the images. On the training images I performed various augmentations. These help to increase the effective size of the dataset. 
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
For the test dataset I only rescalled the images:
```python
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(directory=test_dir, batch_size=20, class_mode='binary', target_size=(224,224))
```

## Modelling
I explored 3 models. A custom CNN, AlexNet and InceptionV3. All models had an input shape of (224, 224, 3). I also defined a callback to terminate training when the training accuracy exceeded 95.0%. This did not happed so training was terminated after the defined number of epochs. Here is my callback function:
```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95.0% accuracy -> Terminating Training")
      self.model.stop_training = True
```
you call this function when you fit the model like so:
```python
history = model.fit(..., callbacks=[callbacks])
```

## Model Performance Comparison

| Model          | Train Accuracy | Test Accuracy |
|----------------|----------------|---------------|
| Custom CNN     | 0.8780         | 0.8700        |
| AlexNet        | 0.6400         | 0.8600        |
| InceptionV3    | 0.8840         | 0.9000        |
