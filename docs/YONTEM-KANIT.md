# Yöntem notu — kanıta dayalı mühendislik ve ratchet felsefesi

Bu belge yöntemi yazar, kod eklemez. Kaynağı ampirik yazılım mühendisliği
(ESE) pratiğidir: **iddia ölçümle taşınır, ölçülen taban yalnız yükselir**.
Buradaki sayılar korpustan türediği için burada yeniden yazılmaz; taşıyıcı
dosyalar `training/ratchet.json` ve `docs/CRATES.md`'dir.

## 1. Ratchet kuralı

Yedi anahtar ölçülür: testler, kapılar, pedantic uyarılar, korpus kayıtları,
korpus jetonları, bootstrap turları, sınav soruları. Dördü **yalnız yükselebilir**,
pedantic **yalnız düşebilir**. Kural mekaniktir: `ratchet-holds` kapısı ölçümü
kendisi yapar ve tabanla karşılaştırır — tabanı yazan ile ölçen aynı kişi
değildir, ölçüm aynı komuttur.

Neden bu kadar katı: bir projede "eskiden daha iyiydi" cümlesi, taban
kaydedilmediği anda ölçülemez olur. Ratchet o cümleyi gereksiz kılar.

## 2. Ölçüm ile iddia ayrılığı

Her kapı iki parçalıdır: **ölçüm** (bir betiği ya da ağacı koşar) ve
**kanarya** (`--self-test`), ki kanarya kapının kendi ihlalini gerçekten
reddettiğini gösterir. Kağıt üstünde kalan bir kural, ihlal eden koşuyu
durdurmuyorsa kural değil dilekçedir (`gate-pairs-carry-referee`).

Aynı çizgi ölçüm kayıtlarında da geçerlidir: bir kaydı elle yazmak yasak;
kaydı üreten betik koşar. Kayıt bayatsa kapı **iki sayıyı birlikte** söyler
("kayıtta 89, ölçüm 138") — böylece hata mesajı, düzeltmenin kendisidir.

## 3. Değişmezler ve damga

Anayasa (`INVARIANTS.md`) donmuştur ve SHA-256 damgası taşır. Damga tutmazsa
döngü başlamaz. Bu, "kural var ama kimse bakmıyor" durumunu mekanik olarak
imkânsız kılar. Kuralı değiştirmek **operatör işidir** ve değişiklik bir commit
olur: söz ile kod ayrışmasın diye makine bloğu tek kaynaktır.

Operatör onayıyla yapılan son değişiklik **K7**'dir (danışma katmanı): dış model
yalnız oy verir, karar kuralı kodda kalır, oy yoksa insan karar verir. Değişiklik
yalnız metin değil, damga yenileme + yeni kapı + kanarya olarak indi.

## 4. Redlerin de ölçülmesi

Sistemin neyi **reddettiği** en az neyi kabul ettiği kadar ölçülür:
enjeksiyon bataryasında red oranı, kimlik taramasında yanlış-pozitif tarafı,
kapsam redlerinde sebep ayrımı (`Revoked` ≠ `NoGrant`). Sıfır ret raporlayan
bir bileşen, denetimlerinin koşmadığını raporluyordur; bu yüzden ret sayısı
kayda geçer.

## 5. Kapsam disiplini

Yeni bir araç ya da bağımlılık, **mevcut çizgiyi zayıflatmıyorsa** girer;
aksi hâlde kaydı bu belgede bir "yapılmadı" maddesi olarak durur (bkz.
`docs/CI-TEHDIT.md` kapsam dışı listesi). Yeşili korumak için eklenen araç,
yeşilin tanımını değiştiriyorsa araç değil tavizdir.

## 6. Uygulama sırası (her commit)

1. düzenle;
2. ölçümü yeniden üret (`build_corpus.py`, ilgili `--kur` betikleri);
3. `python3 gates/check.py --all`;
4. commit; CI tek otoritedir.

Bu sıra 2026-09-24'te yerelde yeşil / CI'da kırmızı sonucu veren bayat ölçüm
hatasını kapatan sıradır: yerel `corpus/` artefaktı bayatken kapı yanlış yeşil
yanıyordu. Ölçüm, ölçülen ağaçla aynı commit'te doğar.
