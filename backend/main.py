"""
YKS Tercih Asistanı — FastAPI backend
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://meryemseymagencer.github.io",  # GitHub Pages adresiniz
        "http://localhost:8000",                # Lokal testler için
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# İstek gövdesi modeli
class MesajModeli(BaseModel):
    rol: str      # "kullanici" veya "asistan"
    metin: str

class SorguIstegi(BaseModel):
    sorgu: str
    gecmis: list[MesajModeli] = []   # opsiyonel; frontend göndermezse boş liste
# ─── API endpoint ───
@app.post("/api/ara")
def ara(istek: SorguIstegi):
    """Kullanıcı sorgusunu alır, asistan sonucunu döndürür."""
    if not istek.sorgu.strip():
        return {"hata": "Boş sorgu", "programlar": [], "yorum": ""}

    # Pydantic modelini asistan.py'nin beklediği düz dict listesine çevir
    gecmis = [{"rol": m.rol, "metin": m.metin} for m in istek.gecmis]

    return asistana_sor(istek.sorgu, gecmis=gecmis)

# ─── Frontend dosyalarını sun ───
# frontend/ klasörünü statik olarak servis et
frontend_yolu = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
def anasayfa():
    return FileResponse(os.path.join(frontend_yolu, "index.html"))

app.mount("/static", StaticFiles(directory=frontend_yolu), name="static")