// ─── Elementler ───
const input = document.getElementById("sorguInput");
const araButon = document.getElementById("araButon");
const yukleniyor = document.getElementById("yukleniyor");
const sonuclar = document.getElementById("sonuclar");

// ─── Arama fonksiyonu ───
async function ara() {
    const sorgu = input.value.trim();
    if (!sorgu) return;

    // UI: yükleniyor durumu
    sonuclar.innerHTML = "";
    yukleniyor.classList.remove("gizli");
    araButon.disabled = true;

    try {
        const cevap = await fetch("/api/ara", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sorgu: sorgu }),
        });

        const veri = await cevap.json();
        sonuclariGoster(veri);
    } catch (hata) {
        sonuclar.innerHTML = `<div class="mesaj">Bir hata oluştu: ${hata.message}</div>`;
    } finally {
        yukleniyor.classList.add("gizli");
        araButon.disabled = false;
    }
}

// ─── Sonuçları ekrana bas ───
function sonuclariGoster(veri) {
    sonuclar.innerHTML = "";

    if (veri.hata || !veri.programlar || veri.programlar.length === 0) {
        sonuclar.innerHTML = `<div class="mesaj">Aradığınız kriterlere uygun program bulunamadı. Farklı bir arama deneyin.</div>`;
        return;
    }

    // GPT yorumu kutusu
    if (veri.yorum) {
        const yorumDiv = document.createElement("div");
        yorumDiv.className = "yorum-kutu";
        yorumDiv.innerHTML = `
            <div class="yorum-baslik">Danışman Notu</div>
            <div>${veri.yorum}</div>
        `;
        sonuclar.appendChild(yorumDiv);
    }

        // Program kartları
    veri.programlar.forEach((p) => {
        const kart = document.createElement("div");
        kart.className = "kart";
        if (p.durum) kart.classList.add("durum-" + p.durum);

        // Detay satırı: sade, tablodakiler tekrar edilmiyor
        let parcalar = [];
        if (p.il) parcalar.push(p.il);
        if (p.dil) parcalar.push(p.dil);
        if (p.tur) parcalar.push(p.tur);
        if (p.puan_turu) parcalar.push(p.puan_turu);   

        // Durum metni (varsa, sonda renkli)
        const durumMetni = {
            girilebilir: "Girilebilir", yeterli: "Puanınız yeterli",
            sinirda: "Sınırda",
            girilemez: "Zorlu", yetmez: "Puan yetersiz",
        };
      
        let durumStr = "";
        if (p.durum && durumMetni[p.durum]) {
            durumStr = ` &nbsp;|&nbsp; <span class="durum-metni durum-${p.durum}">${durumMetni[p.durum]}</span>`;
        }

        // Trend tablosu (son 4 yıl: 2025→2022)
        let trend = "";
        const yillar = [2025, 2024, 2023, 2022];
        const varMi = (p.trend_puan || []).some(x => x != null) || (p.trend_sira || []).some(x => x != null);
        if (varMi) {
            const puanlar = p.trend_puan || [];
            const siralar = p.trend_sira || [];

            let baslikHucreleri = yillar.map(y => `<th>${y}</th>`).join("");
            let puanHucreleri = puanlar.map(x => `<td>${x != null ? x.toFixed(1) : "–"}</td>`).join("");
            let siraHucreleri = siralar.map(x => `<td>${x != null ? Math.round(x).toLocaleString("tr") : "–"}</td>`).join("");

            trend = `
                <div class="kart-trend">
                    <table class="trend-tablo">
                        <thead>
                            <tr><th>Yıl</th>${baslikHucreleri}</tr>
                        </thead>
                        <tbody>
                            <tr><td>Taban puan</td>${puanHucreleri}</tr>
                            <tr><td>Başarı sırası</td>${siraHucreleri}</tr>
                        </tbody>
                    </table>
                </div>
            `;
        }

        kart.innerHTML = `
            <div class="kart-ust">
                <div class="kart-uni">${p.universite}</div>
            </div>
            <div class="kart-bolum">${p.bolum}</div>
            <div class="kart-detay">${parcalar.join(" &nbsp;|&nbsp; ")}${durumStr}</div>
            ${trend}
        `;

        sonuclar.appendChild(kart);
    });
}

// ─── Olay bağlama ───
araButon.addEventListener("click", ara);
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") ara();
});

// Örnek sorgu butonları
document.querySelectorAll(".ornek").forEach((btn) => {
    btn.addEventListener("click", () => {
        input.value = btn.dataset.sorgu;
        ara();
    });
});