import os
import tensorflow
import cv2
import numpy as np

DOGNAMES_TXT_PATH       = os.path.join(os.path.dirname(__file__), "dognames.txt")
MODEL_FOLDER            = os.path.join(os.path.dirname(__file__), "model")
FACEMODEL_XML_PATH      = os.path.join(MODEL_FOLDER, "haarcascade_frontalface_alt.xml")
XCEPTION_HDF5_MODELPATH = os.path.join(MODEL_FOLDER, "Xception.best_weights.hdf5")

with open(DOGNAMES_TXT_PATH, "r") as fd:
    dog_names           = fd.read().splitlines()

def extract_VGG16(tensor):
	from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
	return VGG16(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_VGG19(tensor):
	from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(tensor):
	from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Xception(tensor):
	from tensorflow.keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_InceptionV3(tensor):
	from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    from tensorflow.keras.preprocessing import image
    img                       = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x                         = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def Xception_predictbreed(img_path):
    from tensorflow.keras.models import load_model
    Xception_bottlenecks      = extract_Xception(path_to_tensor(img_path))
    Xception_model            = load_model(XCEPTION_HDF5_MODELPATH)
    Xception_prediction       = Xception_model.predict(Xception_bottlenecks)
    return dog_names[np.argmax(Xception_prediction)]

def ResNet50_predict_labels(img_path):
    # define ResNet50 model
    from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
    ResNet50_model            = ResNet50(weights='imagenet')

    # returns prediction vector for image located at img_path
    img                       = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction                = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):
    img                       = cv2.imread(img_path)
    gray                      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # extract pre-trained face detector
    face_cascade              = cv2.CascadeClassifier(FACEMODEL_XML_PATH)
    faces                     = face_cascade.detectMultiScale(gray)
    return len(faces) > 0