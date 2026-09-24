# Değiştirilemez kurallar (INVARIANTS)

Bu dosya otonom eğitim otomasyonunun anayasasıdır. **Hiçbir bileşen — dahil
kendi kendini geliştirme katmanı — bu dosyaya yazamaz.** Yazma girişimi tek
başına bir durdurma koşuludur (§4/S2) ve kapı `invariants-are-frozen` bunu
mekanik olarak denetler: dosyanın özeti `INVARIANTS.sha256`'da durur, değişen
özet kapıyı kırmızı yakar. Kuralı değiştirmek operatör işidir: operatör dosyayı
düzenler, özeti elle yeniler, değişiklik bir commit olur.

Aşağıdaki blok makine tarafından okunur; otomasyon durdurma koşullarını ve
dokunulmaz dosya listesini buradan alır. Söz ile kod ayrışmasın diye tek
kaynak burasıdır.

```json
{
  "surum": 1,
  "kararlar": [
    {"id": "K1", "kural": "from-scratch base model; hicbir upstream model isimlendirmesi veya agirlik aktarimi yok"},
    {"id": "K2", "kural": "egitim verisi yalnizca Lubot'un kendi repo agacindan gelir; hicbir dis veri kaynagi yok"},
    {"id": "K3", "kural": "buyume yalnizca repo'nun kendi gelisimi ve `doc` komutuyla kabul edilen, record-based provenance'a sahip kayitlar uzerinden olur"},
    {"id": "K4", "kural": "Tier -2 STARK-provable yol yalnizca dogrulama amaclidir; ana egitim planinin parcasi degildir"},
    {"id": "K5", "kural": "operator agreement transition value: zkVM icerik kaniti canli olana kadar 2, sonrasinda 1/1"},
    {"id": "K6", "kural": "egitim, sahibinin kendi donanimiyla sinirlidir; model boyutu donanim kapasitesini asla asamaz"},
    {"id": "no-generation", "kural": "Lubot no-generation-variant bir modeldir: okuyan, uretmeyen; cikti sema dogrulamali Markdown'dir, yumusatma/fallback yoktur"}
  ],
  "dokunulmaz_dosyalar": [
    "autonomous-training/INVARIANTS.md",
    "autonomous-training/INVARIANTS.sha256",
    "autonomous-training/olcut.md",
    "autonomous-training/ayarlar.json",
    "autonomous-training/program.md"
  ],
  "durdurma_kosullari": [
    {"id": "S1", "ad": "insan-karari", "kural": "mimari karar, ekonomik karar, celisen gereksinim ya da somut kapsam catismasi cikarsa dur ve ask_user ile insan onayi bekle"},
    {"id": "S2", "ad": "degistirilemez-kural-yazimi", "kural": "degistirilemez kurallara (bu dosya), durdurma kosullarina ya da degerlendirme kriterinin tanimina yazma girisimi: dur ve insan onayi bekle"},
    {"id": "S3", "ad": "donanim-butcesi-asimi", "kural": "bir deney K6'daki donanim butcesini asan bir model boyutuna dogru gidiyorsa dur"},
    {"id": "S4", "ad": "regresyon-esigi", "kural": "tam dogrulamada basarisiz test ya da kapi sayisi beyan edilen esigi asarsa dur"},
    {"id": "S5", "ad": "kendini-gelistirme-bozulmasi", "kural": "kendi kendini gelistirme katmani art arda beyan edilen sayida deneyde performansi kotulestiren bir strateji degisikligi yaparsa dur"}
  ]
}
```

## Neden bu dosya salt-okunur bir söz değil, ölçülen bir kilit

Bir kural, onu ihlal eden koşuyu durduramıyorsa dilekçedir. Bu yüzden:

1. **Özet kilidi.** `INVARIANTS.sha256` bu dosyanın SHA-256 özetini taşır; kapı
   `invariants-are-frozen` her koşuda özeti yeniden hesaplar. Eşleşmezse
   kapı kırmızıdır ve otomasyon `--tek-tur` bile başlatmaz.
2. **Yazma taraması.** Kapı, otomasyon ağacındaki ve `training/` altındaki kodu
   tarar: dokunulmaz dosyalardan birine yazma çağrısı geçen bir satır varsa
   kırmızı yanar. Söz ile kod ayrışması böyle yakalanır.
3. **Durdurma kanaryası.** `dongu.py --kendini-test` her durdurma koşulunu
   kurar ve otomasyonun *durduğunu* gösterir; durmayan bir koşul kapıyı kırmızı
   yakar. Zararsız durumun durdurmadığı da ayrıca kanıtlanır.
