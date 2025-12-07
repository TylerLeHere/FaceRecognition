# detector.py

from pathlib import Path

import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

"""
Function goes through each directory within training, saves the label from each directory into a name then load the image
"""
def encode_known_faces(
        model: str = "hog", encoding_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        #Use face recognition to detect the face in each image and gets its encoding
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encoding_location.open(mode= "wb") as f:
        pickle.dump(name_encodings, f)



