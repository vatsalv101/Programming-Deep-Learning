from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# 1. Initialize the FastAPI app
app = FastAPI(
    title="ML Model API",
    description="A production-ready API for serving Deep Learning models.",
    version="1.0.0"
)

# 2. Define Data Models using Pydantic
# This ensures data validation. If a user sends bad data, FastAPI automatically returns a 422 Error.
class InferenceRequest(BaseModel):
    text: str
    
class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time_ms: float

# 3. Mock Machine Learning Model
class MockTextClassifier:
    def __init__(self):
        # You would typically load your PyTorch (.pt) or ONNX model here
        print("Model loaded into memory.")

    def predict(self, text: str):
        # Simulate model inference latency
        time.sleep(0.5)
        # Dummy prediction logic
        if "urgent" in text.lower():
            return "High Priority", 0.95
        return "Normal", 0.88

# Initialize model at startup
model = MockTextClassifier()

# 4. Define Endpoints
@app.get("/")
def health_check():
    """Used by systems like Kubernetes or Docker to check if the API is alive."""
    return {"status": "healthy", "message": "API is running."}

@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest):
    """The main inference endpoint."""
    try:
        start_time = time.time()
        
        # Run inference
        prediction, confidence = model.predict(request.text)
        
        process_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=process_time
        )
    except Exception as e:
        # Catch errors so the server doesn't crash
        raise HTTPException(status_code=500, detail=str(e))
