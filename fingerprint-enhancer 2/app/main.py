from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from loguru import logger
import cv2
import numpy as np
from .config import settings
from .enhancer import EnhancerService

app = FastAPI(title=settings.app_name)
service = EnhancerService()

@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}

@app.get("/methods")
def methods():
    return {"methods": service.list_methods()}

@app.post("/enhance")
async def enhance(method: str = Form(...), file: UploadFile = File(...)):
    try:
        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})
        out = service.enhance(img, method)
        ok, buf = cv2.imencode('.png', out)
        if not ok:
            return JSONResponse(status_code=500, content={"error": "Encoding failed"})
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        logger.exception(e)
        return JSONResponse(status_code=500, content={"error": str(e)})
