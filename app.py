import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from third import app as fastapi_app

app = FastAPI(title="AYUSH to ICD Disease Mapping API",
              description="API for mapping AYUSH disease codes to ICD-11 codes using NLP",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the original FastAPI app
app.mount("/api", fastapi_app)

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "AYUSH to ICD Disease Mapping API"}

# Import routes after app is created
from third import *

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
