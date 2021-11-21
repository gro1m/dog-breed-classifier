from dog_breed_classifier.utils import face_detector, dog_detector, Xception_predictbreed

def is_first_letter_a_vocal(input_string):
    vocals = {"A", "E", "I", "O", "U"}
    return any(input_string.startswith(v) for v in vocals)

def predict(img_path):
    is_a_face = face_detector(img_path)                     # detect human face
    is_a_dog = dog_detector(img_path)                       # detect dog
    predicted_dog_breed = Xception_predictbreed(img_path)   # predicted breed

    if is_a_face and not is_a_dog:
        return f"You are human, and the dog breed you resemble most is {predicted_dog_breed}"
    elif not is_a_face and is_a_dog:
        return f"You are an {predicted_dog_breed}" if is_first_letter_a_vocal(predicted_dog_breed) else f"You are a {predicted_dog_breed}"         
    else:
        return f"Well, the dog breed you resemble most is {predicted_dog_breed}"