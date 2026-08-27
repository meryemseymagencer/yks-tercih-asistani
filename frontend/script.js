// ─── Elementler ───
const input = document.getElementById("sorguInput");
const araButon = document.getElementById("araButon");
const yukleniyor = document.getElementById("yukleniyor");
const sonuclar = document.getElementById("sonuclar");
const yeniSohbetButon = document.getElementById("yeniSohbetButon");
const sayfa = document.getElementById("sayfa");
const sohbetKontrol = document.getElementById("sohbetKontrol");
// ─── Sohbet geçmişi (tarayıcıda tutulur, backend stateless kalır) ───
let sohbetGecmisi = [];   // [{rol: "kullanici"/"asistan", metin: "..."}]
// ─── Arama fonksiyonu ───
async function ara() {
    const sorgu = input.value.trim();
    if (!sorgu) return;

    
    araButon.disabled = true;
    const butonYazi = araButon.querySelector(".buton-yazi");
    const eskiYazi = butonYazi.textContent;      // "Ara" veya "Gönder"
    butonYazi.textContent = "…";                 // arama sırasında üç nokta

    try {
        const cevap = await fetch("/api/ara", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sorgu: sorgu, gecmis: sohbetGecmisi }),
        });
        const veri = await cevap.json();
        sonuclariGoster(veri, sorgu);
        input.value = "";   // gönderilince kutuyu boşalt, sonraki soruya hazır
        // Bu turu geçmişe ekle (kullanıcının HAM sorusu + asistanın yorumu)
        sohbetGecmisi.push({ rol: "kullanici", metin: sorgu });
        if (veri.yorum) {
            sohbetGecmisi.push({ rol: "asistan", metin: veri.yorum });
        }

        // Geçmişi son 10 mesajla sınırla (bellek + token kontrolü, backend de kısıyor)
        if (sohbetGecmisi.length > 10) {
            sohbetGecmisi = sohbetGecmisi.slice(-10);
        }

     // Geçmiş oluştuysa sohbet kontrolünü göster + butonu "Gönder" yap
        sohbetKontrol.classList.remove("gizli");
                // İlk sorudan sonra sohbet moduna geç (kutu alta iner, logo küçülür)
        sayfa.classList.add("sohbet-modu");
        araButon.querySelector(".buton-yazi").textContent = "Gönder";

    } catch (hata) {
        sonuclar.innerHTML = `<div class="mesaj">Bir hata oluştu: ${hata.message}</div>`;
    } finally {
        araButon.disabled = false;
        butonYazi.textContent = eskiYazi;         // eski yazıya dön (Ara/Gönder)
    }
}

// ─── Sonuçları ekrana bas ───
function sonuclariGoster(veri, sorgu) {
    // Bu turun tüm çıktısını saracak bir blok oluştur (sohbet akışı için)
    const blok = document.createElement("div");
    blok.className = "tur-blok";

    // Kullanıcının sorusu — sağa yaslı, sade etiket
    if (sorgu) {
        const soruDiv = document.createElement("div");
        soruDiv.className = "kullanici-soru";
        soruDiv.textContent = sorgu;
        blok.appendChild(soruDiv);
    }

    // Sonuç yoksa: bu bloğa mesaj koy, akışa ekle, çık
    if (veri.hata || !veri.programlar || veri.programlar.length === 0) {
        const mesaj = document.createElement("div");
        mesaj.className = "mesaj";
        mesaj.textContent = "Aradığınız kriterlere uygun program bulunamadı. Farklı bir arama deneyin.";
        blok.appendChild(mesaj);
        sonuclar.appendChild(blok);
        blok.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
    }
    // GPT yorumu kutusu (önce)
    if (veri.yorum) {
        const yorumDiv = document.createElement("div");
        yorumDiv.className = "yorum-kutu";
        yorumDiv.innerHTML = `
            <div class="yorum-baslik">Danışman Notu</div>
            <div>${veri.yorum}</div>
        `;
        blok.appendChild(yorumDiv);
    }

    // Kapsam dışı uyarısı (yorumdan sonra)
    if (veri.uyari) {
        const uyariDiv = document.createElement("div");
        uyariDiv.className = "uyari-kutu";
        uyariDiv.textContent = veri.uyari;
        blok.appendChild(uyariDiv);
    }

    // Program kartları
    veri.programlar.forEach((p) => {
        const kart = document.createElement("div");
        kart.className = "kart";
        if (p.durum) kart.classList.add("durum-" + p.durum);

        let parcalar = [];
        if (p.il) parcalar.push(p.il);
        if (p.dil) parcalar.push(p.dil);
        if (p.tur) parcalar.push(p.tur);
        if (p.puan_turu) parcalar.push(p.puan_turu);

        const durumMetni = {
            girilebilir: "Girilebilir", yeterli: "Puan yeterli",
            sinirda: "Sınırda",
            girilemez: "Zorlu", yetmez: "Puan yetersiz",
        };
        let durumStr = "";
        if (p.durum && durumMetni[p.durum]) {
            durumStr = ` &nbsp;|&nbsp; <span class="durum-metni durum-${p.durum}">${durumMetni[p.durum]}</span>`;
        }

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

        blok.appendChild(kart);
    });

    // Bu turun bloğunu sohbet akışına ekle ve oraya kaydır
    sonuclar.appendChild(blok);
    blok.scrollIntoView({ behavior: "smooth", block: "start" });
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
yeniSohbetButon.addEventListener("click", () => {
    sohbetGecmisi = [];
    sonuclar.innerHTML = "";
    input.value = "";
    input.focus();
    sohbetKontrol.classList.add("gizli");
    araButon.querySelector(".buton-yazi").textContent = "Ara";
    sayfa.classList.remove("sohbet-modu");   // karşılama moduna dön
});