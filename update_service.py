from fastapi import FastAPI
import uvicorn
from datetime import datetime

app = FastAPI(title="ERM Update Service")

@app.get("/update")
async def update():
    return {"status": "success", "message": "Endpoint is working", "time": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
