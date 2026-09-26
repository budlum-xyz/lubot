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
    {"id": "K2", "kural": "egitim verisi (a) Lubot'un kendi repo agacindan ve (b) operatore ait ya da kamu mali sinifinda olan dis kaynaklardan gelir; dis kaynagin lisansi indirmeden once kaynagin kendi kaydindan dogrulanir, alinan her parca revizyon ve ozetle (sha256) damgalanir; verinin ne kaynak ne lisans adi depo agacina yazilir"},
    {"id": "K3", "kural": "buyume yalnizca repo'nun kendi gelisimi ve `doc` komutuyla kabul edilen, record-based provenance'a sahip kayitlar uzerinden olur"},
    {"id": "K4", "kural": "Tier -2 STARK-provable yol yalnizca dogrulama amaclidir; ana egitim planinin parcasi degildir"},
    {"id": "K5", "kural": "operator agreement transition value: zkVM icerik kaniti canli olana kadar 2, sonrasinda 1/1"},
    {"id": "K6", "kural": "egitim, sahibinin kendi donanimiyla sinirlidir; model boyutu donanim kapasitesini asla asamaz"},
    {"id": "no-generation", "kural": "Lubot her turden veriyi okuyabilir ve inceleyebilir; kullaniciya sundugu cikti yalnizca sema dogrulamali Markdown'dir. Kullaniciya donuk hicbir yolda yumusatma/fallback yoktur: yapilamayan is acikca reddedilir. Uretilen metin korpusa kayit olamaz; uretim yalnizca kullaniciya donuk yuzeyde olur"},
    {"id": "K7", "kural": "danisma katmani (Jev/Laya) yalnizca OY verir: tut/at karari belirsizlik bandindaysa oy sorulur, esik ve karar kurali kodda kalir, oy yoksa ya da guven esigin altindaysa karar insana gider ve dongu S2 ile durur; danisma katmani olcutu, veriyi, donanimi, degerlendirme tanimini degistiremez, korpusa kayit uretemez"}
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
    {"id": "S2", "ad": "degistirilemez-kural-yazimi", "kural": "degistirilemez kurallara (bu dosya), durdurma kosullarina, degerlendirme kriterinin tanimina ya da danisma kararina yazma/karari devretme girisimi: dur ve insan onayi bekle"},
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
3. **Danışma katmanı sınırı (K7).** Döngü, tut/at kararını *belirsiz* bulduğunda
   (marj, kabul eşiğinin bandı içindeyse) `danisma.py` üzerinden bir **oy** sorar.
   Oy bir tavsiyedir: karar kuralı — eşik, marj, tut/at mantığı — kodda yazılıdır
   ve danışma katmanı onu değiştiremez. Oy gelmezse, geçersizse ya da güveni
   eşiğin altındaysa karar **insana** kalır ve döngü **S2** ile durur. Kapı
   `danisma-layer-is-closed` bu sınırı mekanik olarak denetler: danışma kodunda
   yazma çağrısı yoktur, karar kuralı dosyası dışında marj eşiği hesaplanmaz ve
   kanaryası esik-altı oyun insana gittiğini gösterir.

4. **Durdurma kanaryası.** `dongu.py --kendini-test` her durdurma koşulunu
   kurar ve otomasyonun *durduğunu* gösterir; durmayan bir koşul kapıyı kırmızı
   yakar. Zararsız durumun durdurmadığı da ayrıca kanıtlanır.
