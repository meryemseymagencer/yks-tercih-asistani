"""
YKS Tercih Asistanı - core backend
Parse → filtre → 3 kanal arama → fusion → rerank → formatla → GPT yorum.
"""
import re  
import torch
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
device = "cuda" if torch.cuda.is_available() else "cpu"
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
- basariSirasi_max: sayı. DİKKAT: "300 bin sıralamayla", "ilk 50 bine giren" gibi ifadeler ARALIK bildirir → sayıyı olduğu gibi al (300000, 50000). Ama "Türkiye 300.'süyüm", "5000. sıradayım" gibi ifadeler KESİN SIRA bildirir → o küçük sayıyı al (300, 5000). "300." (nokta ile) sıra demektir, "300 bin" aralık demektir.
- puan_min: sayı (kullanıcının aldığı puan — "450 aldım", "480 puanla girebileceğim")
- ucret_max: sayı (TL)
- ozel_durum: kullanıcının özel kontenjan hakkı ("sehit_gazi", "depremzede", "kktc"). Sadece kullanıcı AÇIKÇA belirtirse ekle.
- kapsam_disi: true/false. Kullanıcı Türkiye DIŞINDA bir ülke veya şehir belirtmişse (Almanya, Londra, Paris, ABD, yurt dışı vb.) true yap. 
Sistem yalnızca Türkiye üniversitelerini kapsıyor.
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
Sorgu: "Türkiye 300.'süyüm, tıp okumak istiyorum"
Çıktı: {"filtreler": {"basariSirasi_max": 300, "birimTuruAdi": "LISANS"}, "arama_metni": "tıp"}

Sorgu: "5000. sıradayım mühendislik"
Çıktı: {"filtreler": {"basariSirasi_max": 5000, "birimTuruAdi": "LISANS"}, "arama_metni": "mühendislik"}

Sorgu: "İstanbul'daki tıp fakülteleri"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "birimTuruAdi": "LISANS"}, "arama_metni": "tıp"}

Sorgu: "Almanya'da mühendislik"
Çıktı: {"filtreler": {}, "arama_metni": "mühendislik", "kapsam_disi": true}

Sorgu: "Londra'da işletme okumak istiyorum"
Çıktı: {"filtreler": {}, "arama_metni": "işletme", "kapsam_disi": true}

Sorgu: "450 puan aldım hangi mühendislik bölümlerine girebilirim"
Çıktı: {"filtreler": {"puan_min": 450, "birimTuruAdi": "LISANS"}, "arama_metni": "mühendislik"}

Sorgu: "İstanbul'da İngilizce bilgisayar mühendisliği"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "ogrenimDiliAdi": "İngilizce"}, "arama_metni": "bilgisayar mühendisliği"}

Sorgu: "300 bin sıralamayla mühendislik"
Çıktı: {"filtreler": {"basariSirasi_max": 300000}, "arama_metni": "mühendislik"}

Sorgu: "KKTC uyrukluyum, İstanbul'da tıp okumak istiyorum"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "birimTuruAdi": "LISANS"}, "arama_metni": "tıp", "ozel_durum": "kktc"}

Sorgu: "Yaratıcılığımı kullanabileceğim bölümler"
Çıktı: {"filtreler": {}, "arama_metni": "yaratıcılık tasarım sanat mimarlık"}

Sorgu: "Şehit yakınıyım, İstanbul'da hukuk okumak istiyorum"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "birimTuruAdi": "LISANS"}, "arama_metni": "hukuk", "ozel_durum": "sehit_gazi"}

Sorgu: "Depremzedeyim, Ankara'da mühendislik"
Çıktı: {"filtreler": {"ilAdi": "ANKARA"}, "arama_metni": "mühendislik", "ozel_durum": "depremzede"}

Sorgu: "İstanbul'da tıp"
Çıktı: {"filtreler": {"ilAdi": "İSTANBUL", "birimTuruAdi": "LISANS"}, "arama_metni": "tıp"}
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
    # Sadece bilinen filtre alanlarını işle
    bilinen = {"ilAdi", "universiteTuru", "ogrenimDiliAdi", "puanTuru",
               "bursOraniAdi", "birimTuruAdi", "basariSirasi_max",
               "puan_min", "ucret_max"}
    kosullar = []
    for alan, deger in filtreler_dict.items():
        if alan not in bilinen:
            continue  # ozel_durum vb. sızarsa atla
        if alan == "basariSirasi_max":
            # Kullanıcının sırasına yakın programlar: sırası [S, S×3] aralığında olanlar.
            # gte=deger  → kullanıcının giremeyeceği daha prestijli okulları (sırası S'den küçük) ele
            # lte=deger*3 → "sıra israfı" olan çok kolay okulları ele (oransal tampon)
            kosullar.append(FieldCondition(
                key="basariSirasi",
                range=Range(gte=deger, lte=deger * 3)
            ))
        elif alan == "puan_min":
                  # Kullanıcının puanına yakın programlar: puanın en fazla 40 altına kadar
                    # (daha düşük taban puanlı programlar "puan israfı" sayılır)
            kosullar.append(FieldCondition( key="minPuan",
                                range=Range(gte=deger - 40, lte=deger)
                            ))
        elif alan == "ucret_max":
            kosullar.append(FieldCondition(key="ucret", range=Range(lte=deger)))
        else:
            kosullar.append(FieldCondition(key=alan, match=MatchValue(value=deger)))
    return Filter(must=kosullar) if kosullar else None
# ─────────────────────────────────────────────
# Çeşitlilik — aynı üniversite/bölümden aşırı tekrarı kırp
# ─────────────────────────────────────────────
def cesitlendir(adaylar, uni_max=2, bolum_max=2):
    uni_sayaci = {}      # her üniversiteden kaç tane aldık
    bolum_sayaci = {}    # her bölüm adından kaç tane aldık
    secilenler = []
    for p in adaylar:
        pl = p.payload
        uni = pl.get("universiteAdi", "")
        # Bölüm adını sadeleştir: parantezli ekleri at ("(Burslu)", "(İngilizce)" vb.)
        bolum = pl.get("birimAdi", "").split("(")[0].strip()

        uni_adet = uni_sayaci.get(uni, 0)
        bolum_adet = bolum_sayaci.get(bolum, 0)

        # İki kotadan biri dolduysa bu adayı atla
        if uni_adet >= uni_max or bolum_adet >= bolum_max:
            continue

        secilenler.append(p)
        uni_sayaci[uni] = uni_adet + 1
        bolum_sayaci[bolum] = bolum_adet + 1
    return secilenler
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
# Her zaman elenecekler (kullanıcının beyan edemeyeceği durumlar)
HER_ZAMAN_ELE = re.compile(r"Yurt\s*Dışından", re.IGNORECASE)

# Özel duruma bağlı elenecekler (kullanıcı beyan ederse gösterilir)
KOSULLU_KALIPLAR = {
    "kktc": re.compile(r"KKTC\s*Uyruklu", re.IGNORECASE),
    "sehit_gazi": re.compile(r"Şehit|Gazi", re.IGNORECASE),
    "depremzede": re.compile(r"Depremzede|Deprem", re.IGNORECASE),
}
# ─────────────────────────────────────────────
# 0) Bağımsızlaştırma — takip sorusunu tek başına anlamlı hale getir
# ─────────────────────────────────────────────
YENIDEN_YAZ_PROMPT = """Görevin, bir üniversite tercih sohbetindeki TAKİP SORUSUNU, \
sohbet geçmişine bakarak TEK BAŞINA anlaşılır bağımsız bir soruya çevirmek.

ÇOK ÖNEMLİ: Soruyu CEVAPLAMA. Üniversite önerme, yorum yapma, bilgi ekleme. \
SADECE soruyu yeniden yaz.

KURALLAR:
- "orası", "oranın", "o bölüm", "peki ya", "aynısı" gibi belirsiz ifadeleri geçmişteki \
somut isimlerle (üniversite, şehir, bölüm) doldur.
- SADECE belirsiz ifadenin işaret ettiği ismi ekle. Geçmişte geçen ama kullanıcının \
sormadığı ek özellikleri (üniversite türü, burs, puan gibi) soruya EKLEME.
- Kullanıcının bu turda yeni eklediği kısıtı (dil, şehir, puan) koru.
- Örnek: geçmiş "İstanbul'da tıp" + asistan "İstanbul Üniversitesi (devlet)" ise, \
takip sorusu "peki İngilizcesi" → "İstanbul'da İngilizce tıp" olmalı. \
"devlet" EKLENMEZ çünkü kullanıcı bunu hiç istemedi.
- Soru zaten bağımsızsa (geçmişe ihtiyaç duymuyorsa) AYNEN geri döndür.
- Geçmişte olmayan bir bilgi UYDURMA. Emin değilsen kullanıcının sorusunu olduğu gibi bırak.
- Çıktı SADECE yeniden yazılmış soru olsun, başka hiçbir şey yazma.

SOHBET GEÇMİŞİ:
{gecmis}

TAKİP SORUSU: {soru}

BAĞIMSIZ SORU:"""

def soruyu_bagimsizlastir(soru: str, gecmis: list | None = None):
    # Geçmiş yoksa/boşsa: bu ilk soru, bağımsızlaştırmaya gerek yok.
    # Ekstra LLM çağrısı yapmadan soruyu olduğu gibi döndür.
    if not gecmis:
        return soru

    # Geçmişi düz metne çevir. gecmis = [{"rol": "kullanici"/"asistan", "metin": "..."}]
    satirlar = []
    for m in gecmis[-10:]:                      # en fazla son 10 mesaj (hocanın sınırı)
        rol = "Kullanıcı" if m["rol"] == "kullanici" else "Asistan"
        metin = m["metin"]
        if m["rol"] == "asistan":
            metin = metin[:200]                 # asistan cevabını kısalt (token tasarrufu)
        satirlar.append(f"{rol}: {metin}")
    gecmis_metni = "\n".join(satirlar)

    prompt = YENIDEN_YAZ_PROMPT.replace("{gecmis}", gecmis_metni).replace("{soru}", soru)

    try:
        r = openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,                       # deterministik olsun, yaratıcılık istemiyoruz
        )
        bagimsiz = r.choices[0].message.content.strip()
        return bagimsiz if bagimsiz else soru    # boş dönerse orijinale düş
    except Exception as e:
        print(f"Bağımsızlaştırma hatası: {e}")
        return soru                              # hata olursa orijinal soruyla devam et
def arama_yap(kullanici_sorgusu: str, limit: int = 5, aday_sayisi: int = 20, gecmis: list | None = None):
    # ─── ADIM 0: Bağımsızlaştırma (takip sorusuysa) ───
    orijinal_sorgu = kullanici_sorgusu
    kullanici_sorgusu = soruyu_bagimsizlastir(kullanici_sorgusu, gecmis)

    # ─── DEBUG 0: Bağımsızlaştırma ───
    if gecmis:  # sadece geçmiş varsa bas (ilk soruda gürültü olmasın)
        print("\n" + "="*70)
        print(f"[0] BAĞIMSIZLAŞTIRMA (geçmiş: {len(gecmis)} mesaj):")
        print(f"    orijinal : {orijinal_sorgu!r}")
        print(f"    bağımsız : {kullanici_sorgusu!r}")
        if orijinal_sorgu == kullanici_sorgusu:
            print(f"    → değişmedi (soru zaten bağımsızdı)")
    parse = sorguyu_parse_et(kullanici_sorgusu)
    if parse is None:
        return None, None
    
    # ─── DEBUG 1: Parse ───
    print("\n" + "="*70)
    print(f"SORGU: {kullanici_sorgusu!r}")
    print("="*70)
    print(f"[1] PARSE:")
    print(f"    filtreler   : {parse.get('filtreler', {})}")
    print(f"    arama_metni : {parse.get('arama_metni')!r}")
    print(f"    ozel_durum  : {parse.get('ozel_durum')}")

    qfilter = filtre_olustur(parse.get('filtreler', {}))
    print(f"[2] QDRANT FİLTRE: {'var' if qfilter else 'yok'}")
    if qfilter:
        for k in qfilter.must:
            print(f"    - {k.key}: match={getattr(k.match,'value',None) if k.match else None} "
                  f"range={f'gte={k.range.gte},lte={k.range.lte}' if k.range else None}")

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

    # ─── DEBUG 3: Fusion (rerank öncesi) ───
    print(f"[3] FUSION SONUCU (rerank öncesi, {len(sonuclar.points)} aday):")
    for i, p in enumerate(sonuclar.points[:20], 1):
        pl = p.payload
        print(f"    {i:2}. [{p.score:.4f}] {pl.get('birimAdi')} — {pl.get('universiteAdi')} ({pl.get('ilAdi')})")

    # Özel kontenjan elemesi
    kullanici_durumu = parse.get("ozel_durum")

    def elenmeli(birim_adi):
        birim_adi = birim_adi or ""
        if HER_ZAMAN_ELE.search(birim_adi):
            return True
        for durum, kalip in KOSULLU_KALIPLAR.items():
            if kalip.search(birim_adi) and kullanici_durumu != durum:
                return True
        return False

    temiz_sonuclar = [p for p in sonuclar.points if not elenmeli(p.payload.get("birimAdi"))]
    elenen_sayi = len(sonuclar.points) - len(temiz_sonuclar)
    print(f"[4] ÖZEL KONTENJAN ELEMESİ: {elenen_sayi} program elendi "
          f"({len(temiz_sonuclar)} kaldı)")

    # ─── Çeşitlilik: aynı üni/bölümden aşırı tekrarı kırp (rerank öncesi) ───
    cesitli_sonuclar = cesitlendir(temiz_sonuclar, uni_max=2, bolum_max=2)
    print(f"[4.5] ÇEŞİTLİLİK FİLTRESİ: {len(temiz_sonuclar)} → {len(cesitli_sonuclar)} aday "
          f"(her üni/bölümden max 2)")

    rerank_sonuc = rerank_et(parse['arama_metni'], temiz_sonuclar, top_k=limit)

    # ─── DEBUG 5: Rerank sonucu ───
    print(f"[5] RERANK SONUCU (içerik bazlı yeniden sıralama):")
    for i, p in enumerate(rerank_sonuc, 1):
        pl = p.payload
        print(f"    {i}. [rerank={p.score:.4f}] {pl.get('birimAdi')} — {pl.get('universiteAdi')}")

    # Puana göre sırala
    rerank_sonuc = sorted(
        rerank_sonuc,
        key=lambda p: p.payload.get("minPuan") or 0,
        reverse=True,
    )

    # ─── DEBUG 6: Puan sıralaması (son hali) ───
    print(f"[6] PUANA GÖRE SIRALI (kullanıcıya gidecek):")
    for i, p in enumerate(rerank_sonuc, 1):
        pl = p.payload
        puan = pl.get('minPuan')
        print(f"    {i}. {pl.get('universiteAdi')} — {pl.get('birimAdi')} "
              f"(puan: {puan if puan else '—'})")
    print("="*70 + "\n")

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
#bu fonksiyon, modelin yorum üretmesi için veriyi temiz ve okunabilir bir metin haline getirir

# ─────────────────────────────────────────────
# 7) GPT yorumu
# ─────────────────────────────────────────────
CEVAP_PROMPT = """# ROLE
You are an experienced university admissions advisor for Turkish students. You guide students toward the right choice based on YÖK Atlas (Turkish Higher Education) data, using a warm and reassuring tone. You never state that you are an AI model; you speak like a real advisor.

# TASK
Review the user's question and the programs found by the system, then write a short, friendly comment of 3-4 sentences. Your comment should make the user's decision easier.

# INPUTS
User question:
{soru}

Programs found (ordered by score):
{programlar}

# RULES
- Use ONLY the information in the "Programs found" list above. Never invent any university, department, score, or ranking outside this list.
- Never change numbers or names; stay faithful to exactly what the list says.
- Evaluate programs from highest score to lowest (best choice to lesser choice).
- If asked about something unknown or absent from the list, admit it honestly (e.g., "There is no matching result in my current data").
- If the user's question does not match the programs found in meaning (for example, a foreign country is asked but the list contains Turkish programs), point this out politely and state that the list shows the closest available results.
- Do not make exaggerated promises (never say "you will definitely get in"); be realistic and balanced.
- If you dont have enough information to give a confident answer, say so. Avoid making assumptions or guesses.

# WRITING STRUCTURE
- Do NOT open with praise or affirmations. Never begin with phrases like "harika bir seçim", "harika bir adım", "güzel bir hedef" or similar. These sound artificial and repetitive.
- Opening: Start directly with the most relevant, concrete observation about the programs found (e.g. which program stands out and why).
- Body: A concrete, data-driven assessment of 1-2 standout programs from the list.
- Closing: A brief, natural closing that helps the user decide — without empty encouragement clichés.

# OUTPUT LANGUAGE
Write your entire response in TURKISH. The instructions above are in English, but the final comment shown to the user must be in fluent, natural Turkish.

# EXPECTED OUTPUT
Only the comment text in Turkish. Do NOT use any title, bullet points, or prefixes like "Here is the comment:". Speak directly as an advisor, in flowing 3-4 sentences.

Yorum:"""
def yorum_uret(kullanici_sorgusu, programlar, kapsam_disi=False):
    metin = programlari_metne_cevir(programlar)
    prompt = CEVAP_PROMPT.replace("{soru}", str(kullanici_sorgusu)).replace("{programlar}", metin)

    # Kapsam dışıysa GPT'ye özel talimat ekle
    if kapsam_disi:
        prompt += ("\n\n# ÖNEMLİ: Kullanıcı Türkiye dışında bir konum belirtti. "
                   "Sistem yalnızca Türkiye üniversitelerini kapsıyor. Yorumunda o yurt dışı "
                   "konumdan OLUMLU şekilde bahsetme, 'harika hedef' gibi ifadeler kullanma. "
                   "Nazikçe yalnızca Türkiye programları sunabileceğini belirt ve listedeki "
                   "Türkiye programlarına odaklan.")

    r = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=250,
    )
    return r.choices[0].message.content
# ─────────────────────────────────────────────
# 8) Ana giriş noktası — API bunu çağıracak
# ─────────────────────────────────────────────
def asistana_sor(kullanici_sorgusu, gecmis: list | None = None):
    sonuclar, parse = arama_yap(kullanici_sorgusu, limit=5, gecmis=gecmis)
    if sonuclar is None:
        return {"hata": "Sorgu anlaşılamadı", "programlar": [], "yorum": ""}

    filtreler = parse.get('filtreler', {})
    hedef_sira = filtreler.get('basariSirasi_max')
    hedef_puan = filtreler.get('puan_min')
    programlar = programlari_formatla(sonuclar, hedef_sira=hedef_sira, hedef_puan=hedef_puan)
    yorum = yorum_uret(kullanici_sorgusu, programlar, kapsam_disi=parse.get("kapsam_disi", False))
    sonuc = {
        "sorgu": kullanici_sorgusu,
        "parse": parse,
        "programlar": programlar,
        "yorum": yorum,
    }

    # Kapsam dışı uyarısı
    if parse.get("kapsam_disi"):
        sonuc["uyari"] = ("Sistemimiz yalnızca Türkiye'deki üniversite programlarını "
                          "kapsamaktadır. Belirttiğiniz yurt dışı konum için sonuç veremiyoruz. "
                          "Aşağıda Türkiye'deki en alakalı programları görebilirsiniz.")

    return sonuc