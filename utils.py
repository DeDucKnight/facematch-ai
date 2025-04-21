from deepface import DeepFace
from typing import Tuple
from fastapi import UploadFile
import shutil
import uuid
import os
from typing import List


def save_temp_image(upload_file: UploadFile) -> str:
    """
    Save uploaded image to a temporary file and return its path.
    """
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_filename


def compare_faces_deepface(file1: UploadFile, file2: UploadFile) -> Tuple[bool, float]:
    """
    Compares two faces using DeepFace and returns (match, distance/confidence).
    """
    # Save files temporarily
    path1 = save_temp_image(file1)
    path2 = save_temp_image(file2)

    try:
        result = DeepFace.verify(
            img1_path=path1, 
            img2_path=path2, 
            model_name="Facenet", 
            detector_backend="retinaface",
            threshold=0.6,
            enforce_detection=True
        )

        # Print the result for debugging
        print("Model:", result["model"])
        print("Distance:", result["distance"])
        print("Threshold:", result["threshold"])
        print("Verified:", result["verified"])

        # match = result["verified"]
        distance = result["distance"]
        threshold = 0.6 # Adjust threshold as needed
        match = distance < threshold # Manually override the match condition
        return match, distance
    finally:
        # Clean up temp files
        os.remove(path1)
        os.remove(path2)

def extract_embedding(file: UploadFile, model_name="Facenet", detector_backend="retinaface") -> List[float]:
    """
    Extracts face embedding from a single uploaded image.
    Returns a list of floats (embedding vector).
    """
    path = save_temp_image(file)

    try:
        embedding_objs = DeepFace.represent(
            img_path=path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        if not embedding_objs:
            raise ValueError("No face detected.")
        
        return embedding_objs[0]["embedding"]
    finally:
        os.remove(path)

