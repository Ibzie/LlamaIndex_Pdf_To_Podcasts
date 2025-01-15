from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path
from .main import main

app = FastAPI()
root_dir = Path(__file__).parent.parent

@app.post("/generate-podcast/")
async def generate_podcast(file: UploadFile = File(...)):
    try:
        input_path = os.path.join(root_dir, "Data/input.pdf")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        main()
        
        episodes_dir = os.path.join(root_dir, "Data/podcast_episodes")
        if os.path.exists(episodes_dir):
            episodes = os.listdir(episodes_dir)
            if episodes:
                return FileResponse(
                    os.path.join(episodes_dir, episodes[0]),
                    media_type="audio/mpeg",
                    filename="podcast.mp3"
                )
        
        return {"error": "No audio generated"}
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)