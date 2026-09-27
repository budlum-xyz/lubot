# Grant defteri — standart IAM desenleriyle kıyas (karar kaydı)

Bu belge **kod değil karar kaydıdır**: `crates/grant` grant defterini yerleşik
IAM / OpenID Connect desenleriyle yan yana koyar, eşleşenleri ve **bilerek
eşleşmeyenleri** gerekçesiyle yazar. Amaç "olgunluk puanı" değil, farkların
nerede olduğunu ve neden orada olduğunu kayda geçirmek. Ölçülen sayılar burada
yeniden yazılmaz (korpustan türeyen sayı geri beslenir); kaynak kanıtı
`crates/grant` testleri ve `docs/CRATES.md`'dir.

## Eşleşen desenler

| desen (standart) | lubot karşılığı | neden aynı |
|---|---|---|
| **Capability / bearer token** | görüntüleme izni: `grantee` + `content_key_id` + bitiş | Yetki, kimliğe değil *kaynağa* bağlı bir yetenektir; taşıyıcı kaynağı açabilir, başkasını açamaz. |
| **Expiry (TTL)** | izin bitiş anı taşır | Süre biten yetki sessizce uzamaz; fail-closed taraftır. |
| **Revocation list** | `Decision::Revoked` | Geri alma yeni açmaları durdurur. |
| **Audit log, izin ve ret aynı biçimde** | denetim günlüğü | "Sıfır ret" raporlayan dağıtım, denetimlerinin koşmadığını raporluyordur. |
| **Fail-closed karar** | bilinmeyen durumda açmama | Varsayılan cevap "hayır"; izin açık bir kayıttır. |
| **Least privilege / kapsam** | kapsam redleri (üretim, gizli avı) | İzin ne kadar geniş olursa olsun, üretim yüzeyi yoktur. |

## Bilerek eşleşmeyen desenler

| desen (standart) | lubot'ta | gerekçe |
|---|---|---|
| **Refresh token / sessiz yenileme** | **yok** | Sessiz yenileme, süresi geçmiş bir yetkiyi canlı tutar; burada bitiş *karardır*, öneri değil. Yenileme istisna değil, yeni izindir. |
| **Kimlik sağlayıcı (IdP) / token introspection** | **yok (kapsam dışı)** | Zincir kimliği ve operatör kaydı düğümün işidir; Lubot bir *istemci*dir. Kendi kimlik doğrulayıcısını icat etmesi, zincirin tek otorite olması kuralını bozardı. |
| **Scope suffix / wildcard (`resource:*`)** | **yok** | Joker kapsam, "neyi vermediğini" belirsizleştirir. İzin tek içerik anahtarına bağlıdır. |
| **Rol devralma (assume-role) zinciri** | **yok** | Yetki devredilemez; devretmek geri almayı izlenemez kılar. |
| **Merkezi PDP/PEP ayrımı** | **kısmen**: karar `crates/grant`, uygulama `crates/answer` | Ayrım var ama dağıtık değil; tek süreç içinde kalır, çünkü ölçek buna ihtiyaç duymuyor. "Olmayan bir dağıtımı taklit etmek" yerine sade olan yazıldı. |
| **mTLS / imzalı token (JWS)** | **kısmen**: zincir kaydı ve özet doğrulaması | Baytlar ve kayıtlar özetle doğrulanır; izin kaydının kendisi imzalı taşıyıcı değildir, çünkü izin *yereldir* ve ağ üzerinden taşınmaz. |

## Bilinçli asimetri: geri alma geçmişi geri getirmez

Standart IAM'de de böyledir ama burada **açıkça** karardır: `Revoked`, `NoGrant`
değildir. İkisini tek cevaba indirmek, "zaten hiç izin verilmemişti" demek
olurdu; bu geçmiş hakkında yalan olurdu. Kıyasın en taşıyıcı maddesi budur:
**ret sebepleri ayrıştırılır, çünkü sebep sonradan denetlenir.**

## Açık iş

- Grant defterinin standart **karar noktası** adlarıyla (`PDP/PEP`, `RAR`,
  `token exchange`) eşlenmesi bir *doküman* işi olarak bu belgede duruyor;
  kod ekleme gerektirmiyor.
- Zincir tarafında `TrainingDataGrant` (zaman + azami epoch) vardır;
  görüntüleme izninin zincirle kayıt düzeyinde bağlanması gelecek iştir
  (`SECURITY.md`'de kayıtlı).
