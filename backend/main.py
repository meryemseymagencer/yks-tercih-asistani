"""
YKS Tercih Asistanı — FastAPI backend.
Frontend'den sorgu alır, asistan.py'yi çağırır, JSON döndürür.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from asistan import asistana_sor

app = FastAPI(title="YKS Tercih Asistanı")

# Frontend'in backend'e istek atabilmesi için CORS izni
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # geliştirme için hepsi; prod'da kısıtlanır
    allow_methods=["*"],
    allow_headers=["*"],
)


# İstek gövdesi modeli
class SorguIstegi(BaseModel):
    sorgu: str


# ─── API endpoint ───
@app.post("/api/ara")
def ara(istek: SorguIstegi):
    """Kullanıcı sorgusunu alır, asistan sonucunu döndürür."""
    if not istek.sorgu.strip():
        return {"hata": "Boş sorgu", "programlar": [], "yorum": ""}
    return asistana_sor(istek.sorgu)


# ─── Frontend dosyalarını sun ───
# frontend/ klasörünü statik olarak servis et
frontend_yolu = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def anasayfa():
    return FileResponse(os.path.join(frontend_yolu, "index.html"))

app.mount("/static", StaticFiles(directory=frontend_yolu), name="static")