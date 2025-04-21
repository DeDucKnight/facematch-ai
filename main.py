from fastapi import FastAPI, UploadFile, File, HTTPException
from utils import compare_faces_deepface, extract_embedding

# Dummy user database for demonstration purposes
user_db = {}

app = FastAPI()


@app.post("/match")
async def match_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        match, distance = compare_faces_deepface(image1, image2)
        return {
            "match": match,
            "distance": round(distance, 4),
            "message": "Faces match!" if match else "Faces do not match."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/enroll")
async def enroll_user(user_id: str, image: UploadFile = File(...)):
    try:
        embedding = extract_embedding(image)
        user_db[user_id] = embedding
        return {"message": f"User '{user_id}' enrolled successfully.", "vector_length": len(embedding)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
