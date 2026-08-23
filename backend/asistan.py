"""
YKS Tercih Asistanı - core backend
Parse → filtre → 3 kanal arama → fusion → rerank → formatla → GPT yorum.
"""
import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Range,
    Prefetch, FusionQuery, Fusion, Document, PayloadSchemaType,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from openai import OpenAI

# ─────────────────────────────────────────────
# Kurulum
# ─────────────────────────────────────────────
load_dotenv()

KOLEKSIYON = "yks_named"
GPT_MODEL = "gpt-4o-mini"
device = "cpu"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    port=443, https=True, timeout=60, prefer_grpc=False,
)

print("Modeller yükleniyor...")
model = SentenceTransformer("BAAI/bge-m3", device=device)
reranker = AutoModel.from_pretrained(
    "jinaai/jina-reranker-v3.5", dtype="auto", trust_remote_code=True,
)
reranker.eval()
reranker.to(device)
print(f"Hazır — cihaz: {device}, koleksiyon: {KOLEKSIYON}")


# ─────────────────────────────────────────────
# Sayısal alan index'leri (Range filtresi için, tek seferlik)
# ─────────────────────────────────────────────
def index_kur():
    for alan in ["minPuan", "basariSirasi", "ucret"]:
        try:
            qdrant.create_payload_index(
                collection_name=KOLEKSIYON,
                field_name=alan,
                field_schema=PayloadSchemaType.FLOAT,
            )
        except Exception:
            pass  # zaten varsa geç

index_kur()


# ─────────────────────────────────────────────
# 1) Parser — doğal dil → yapılandırılmış JSON
# ─────────────────────────────────────────────
PARSER_PROMPT = """Sen bir üniversite tercih asistanının sorgu ayrıştırıcısısın.
Kullanıcının doğal dildeki sorgusunu, veritabanı araması için yapılandırılmış JSON'a çevir.

MEVCUT FİLTRE ALANLARI:
- ilAdi: "İSTANBUL", "ANKARA", "İZMİR", "KOCAELİ", "BURSA" vb. (BÜYÜK HARF)
- universiteTuru: "DEVLET" veya "VAKIF"
- ogrenimDiliAdi: "Türkçe", "İngilizce", "İngilizce (%30)", "Almanca" vb.
- puanTuru: "SAY", "SÖZ", "EA", "DİL", "TYT"
- bursOraniAdi: "Burslu", "%50 İndirimli", "%25 İndirimli", "Ücretli"
- birimTuruAdi: "LISANS" (4 yıllık) veya "ÖNLISANS" (2 yıllık)
- basariSirasi_max: sayı (üst sınır — "300 bin sıralamayla" veya "ilk 10 bine giren")
- puan_min: sayı (kullanıcının aldığı puan — "450 aldım", "480 puanla girebileceğim")
- ucret_max: sayı (TL)

ÇIKTI FORMATI (JSON):
{
  "filtreler": { alan: değer, ... },
  "arama_metni": "vektör aramada kullanılacak metin"
}

KURALLAR:
- Sadece sorguda AÇIKÇA geçen filtreleri ekle
- Şehir isimlerini BÜYÜK HARF yaz
- Bölüm adları (tıp, mühendislik, hukuk vb.) filtre DEĞİL, arama_metni'ne yaz
- Bilinmeyen alan uydurma

AKILLI ÇIKARIM (birimTuruAdi için):
- "tıp", "hukuk", "mühendislik", "eczacılık", "diş hekimliği", "mimarlık", "psikoloji" → LISANS
- "teknikerlik", "operatörlük", "yardımcılığı" → ÖNLISANS

ÖRNEKLER:

Sorgu: "İstanbul'daki tıp fakülteleri"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "birimTuruAdi": "LISANS"}, "arama_metni": "tıp"}

Sorgu: "450 puan aldım hangi mühendislik bölümlerine girebilirim"
Çıktı: {"filtreler": {"puan_min": 450, "birimTuruAdi": "LISANS"}, "arama_metni": "mühendislik"}

Sorgu: "İstanbul'da İngilizce bilgisayar mühendisliği"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "ogrenimDiliAdi": "İngilizce"}, "arama_metni": "bilgisayar mühendisliği"}

Sorgu: "300 bin sıralamayla mühendislik"
Çıktı: {"filtreler": {"basariSirasi_max": 300000}, "arama_metni": "mühendislik"}

Sorgu: "Yaratıcılığımı kullanabileceğim bölümler"
Çıktı: {"filtreler": {}, "arama_metni": "yaratıcılık tasarım sanat mimarlık"}

Şimdi sıradaki sorguyu ayrıştır:

Sorgu: "{kullanici_sorgusu}"
Çıktı:"""


def sorguyu_parse_et(kullanici_sorgusu: str):
    prompt = PARSER_PROMPT.replace("{kullanici_sorgusu}", kullanici_sorgusu)
    try:
        r = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print(f"Parse hatası: {e}")
        return None


# ─────────────────────────────────────────────
# 2) Filtre — parse dict'inden Qdrant Filter
# ─────────────────────────────────────────────
def filtre_olustur(filtreler_dict: dict):
    if not filtreler_dict:
        return None
    kosullar = []
    for alan, deger in filtreler_dict.items():
        if alan == "basariSirasi_max":
        # Kullanıcı 300.000. sırada → taban sırası >= 300.000 olan programlara girer
            kosullar.append(FieldCondition(key="basariSirasi", range=Range(gte=deger)))
        elif alan == "puan_min":
            kosullar.append(FieldCondition(key="minPuan", range=Range(lte=deger)))
        elif alan == "ucret_max":
            kosullar.append(FieldCondition(key="ucret", range=Range(lte=deger)))
        else:
            kosullar.append(FieldCondition(key=alan, match=MatchValue(value=deger)))
    return Filter(must=kosullar) if kosullar else None


# ─────────────────────────────────────────────
# 3) Reranker — Jina listwise
# ─────────────────────────────────────────────
def rerank_et(arama_metni: str, sonuclar, top_k: int = 5):
    if not sonuclar:
        return sonuclar
    dokumanlar = []
    for p in sonuclar:
        pl = p.payload
        metin = f"{pl.get('birimAdi','')} - {pl.get('universiteAdi','')} - {pl.get('ilAdi','')} - {pl.get('ogrenimDiliAdi','')}"
        dokumanlar.append(metin)

    sonuc = reranker.rerank(arama_metni, dokumanlar, top_n=top_k)
    yeniden_sirali = []
    for r in sonuc:
        orijinal = sonuclar[r['index']]
        orijinal.score = float(r['relevance_score'])
        yeniden_sirali.append(orijinal)
    return yeniden_sirali


# ─────────────────────────────────────────────
# 4) Arama — 3 kanal fusion + rerank
# ─────────────────────────────────────────────
def arama_yap(kullanici_sorgusu: str, limit: int = 5, aday_sayisi: int = 20):
    parse = sorguyu_parse_et(kullanici_sorgusu)
    if parse is None:
        return None, None

    qfilter = filtre_olustur(parse['filtreler'])
    q_vek = model.encode(parse['arama_metni'], normalize_embeddings=True).tolist()

    sonuclar = qdrant.query_points(
        collection_name=KOLEKSIYON,
        prefetch=[
            Prefetch(query=q_vek, using="title", limit=30, filter=qfilter),
            Prefetch(query=q_vek, using="content", limit=30, filter=qfilter),
            Prefetch(query=Document(text=parse['arama_metni'], model="Qdrant/bm25"),
                     using="bm25", limit=30, filter=qfilter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=aday_sayisi,
        with_payload=[
            "universiteAdi", "birimAdi", "ilAdi",
            "ogrenimDiliAdi", "puanTuru", "universiteTuru",
            "bursOraniAdi", "ucret",
            "minPuan", "basariSirasi",
            "minPuan1", "basariSirasi1",
            "minPuan2", "basariSirasi2",
            "minPuan3", "basariSirasi3",
        ],
    )
    rerank_sonuc = rerank_et(parse['arama_metni'], sonuclar.points, top_k=limit)
    return rerank_sonuc, parse


# ─────────────────────────────────────────────
# 5) Formatla — deterministik program listesi
# ─────────────────────────────────────────────
def programlari_formatla(sonuclar, hedef_sira=None, hedef_puan=None):
    programlar = []
    for i, p in enumerate(sonuclar, 1):
        pl = p.payload
        program = {
            "sira_no": i,
            "universite": pl.get('universiteAdi', '?'),
            "bolum": pl.get('birimAdi', '?'),
            "il": pl.get('ilAdi', '?'),
            "dil": pl.get('ogrenimDiliAdi'),
            "tur": pl.get('universiteTuru'),
            "puan_turu": pl.get('puanTuru'),
            "basari_sirasi": int(pl['basariSirasi']) if pl.get('basariSirasi') else None,
            "min_puan": round(pl['minPuan'], 2) if pl.get('minPuan') else None,
            "trend_puan": [pl.get('minPuan'), pl.get('minPuan1'), pl.get('minPuan2'), pl.get('minPuan3')],
            "trend_sira": [pl.get('basariSirasi'), pl.get('basariSirasi1'), pl.get('basariSirasi2'), pl.get('basariSirasi3')],
        }
        # Değerlendirme etiketi
        if hedef_sira and program["basari_sirasi"]:
            s = program["basari_sirasi"]
            # Program taban sırası, kullanıcının sırasından büyük/eşitse girilebilir
            program["durum"] = "girilebilir" if s >= hedef_sira else ("sinirda" if s >= hedef_sira * 0.90 else "girilemez")
        elif hedef_puan and program["min_puan"]:
            program["durum"] = "yeterli" if program["min_puan"] <= hedef_puan else ("sinirda" if program["min_puan"] <= hedef_puan+10 else "yetmez")
        else:
            program["durum"] = None
        programlar.append(program)
    return programlar


# ─────────────────────────────────────────────
# 6) Metin formatı (GPT'ye vermek için)
# ─────────────────────────────────────────────
def programlari_metne_cevir(programlar):
    satirlar = []
    for p in programlar:
        s = f"{p['sira_no']}. {p['universite']} — {p['bolum']} ({p['il']}, {p['dil']}, {p['tur']})"
        if p['basari_sirasi']:
            s += f" | sıra: {p['basari_sirasi']}"
        if p['min_puan']:
            s += f" | min puan: {p['min_puan']}"
        satirlar.append(s)
    return "\n".join(satirlar)


# ─────────────────────────────────────────────
# 7) GPT yorumu
# ─────────────────────────────────────────────
CEVAP_PROMPT = """Sen bir üniversite tercih danışmanısın. Kullanıcının sorusu ve
sistemin bulduğu programlar aşağıda. 3-4 cümlelik kısa, samimi bir yorum yaz.
Sayıları/isimleri değiştirme, verilen listeye sadık kal.

Kullanıcı sorusu: {soru}

Bulunan programlar:
{programlar}

Kısa yorum:"""


def yorum_uret(kullanici_sorgusu: str, programlar):
    metin = programlari_metne_cevir(programlar)
    prompt = CEVAP_PROMPT.replace("{soru}", kullanici_sorgusu).replace("{programlar}", metin)
    r = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250,
    )
    return r.choices[0].message.content


# ─────────────────────────────────────────────
# 8) Ana giriş noktası — API bunu çağıracak
# ─────────────────────────────────────────────
def asistana_sor(kullanici_sorgusu: str):
    sonuclar, parse = arama_yap(kullanici_sorgusu, limit=5)
    if sonuclar is None:
        return {"hata": "Sorgu anlaşılamadı", "programlar": [], "yorum": ""}

    hedef_sira = parse['filtreler'].get('basariSirasi_max')
    hedef_puan = parse['filtreler'].get('puan_min')
    programlar = programlari_formatla(sonuclar, hedef_sira=hedef_sira, hedef_puan=hedef_puan)
    yorum = yorum_uret(kullanici_sorgusu, programlar)

    return {
        "sorgu": kullanici_sorgusu,
        "parse": parse,
        "programlar": programlar,
        "yorum": yorum,
    }