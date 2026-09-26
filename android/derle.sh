#!/usr/bin/env bash
# Lubot APK'sını Gradle olmadan kurar.
#
# Gradle/AGP bilerek yok: Java 11 ile çalışan bir zincir istiyoruz ve AGP 8
# Java 17 istiyor. Onun yerine Android'in kendi araçları kullanılıyor —
# aapt2 kaynakları derler, javac android.jar'a karşı derler, d8 dex yapar,
# apksigner imzalar. Her adım görünebilir ve tek başına yeniden koşulabilir.
#
# Kullanım: ANDROID_SDK=/yol android/derle.sh
set -euo pipefail

KOK="$(cd "$(dirname "$0")/.." && pwd)"
ANDROID="${ANDROID_SDK:-/home/user/android-sdk}"
BT="$ANDROID/android-14"                 # build-tools r34
NDK="$ANDROID/android-ndk-r27c"
PLAT="$ANDROID/android-34"
HARF="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
API=24
ABI=arm64-v8a
TRIPLE=aarch64-linux-android
CIKTI="$KOK/android/out"
PKG_YOL=dev/budlum/lubot

export CARGO_HOME="${CARGO_HOME:-/home/user/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/home/user/.rustup}"
# JDK araçları (keytool dahil) alternatif symlink'lerinde olmayabiliyor.
JDK_BIN="${JAVA_HOME:-/usr/lib/jvm/jdk-11}/bin"
export PATH="$CARGO_HOME/bin:$HARF/bin:$BT:$JDK_BIN:$PATH"

echo "== 0/8 kontrol =="
for a in aapt2 d8 zipalign apksigner javac cargo rustc; do
  command -v "$a" >/dev/null || { echo "eksik araç: $a"; exit 1; }
done
test -f "$PLAT/android.jar" || { echo "android.jar yok: $PLAT"; exit 1; }

rm -rf "$CIKTI"
mkdir -p "$CIKTI/gen" "$CIKTI/obj" "$CIKTI/dex"
mkdir -p "$CIKTI/stage/lib/$ABI" "$CIKTI/stage/assets/korpus"

echo "== 1/8 Rust hedefi =="
rustup target add "$TRIPLE" >/dev/null 2>&1 || true
rustc --print target-list | grep -q "^$TRIPLE$" || { echo "hedef yok: $TRIPLE"; exit 1; }

echo "== 2/8 Rust cdylib ($ABI) =="
# Cargo hedef değişkenlerinde tire olmaz: AARCH64-LINUX-ANDROID -> AARCH64_LINUX_ANDROID
TRIPLE_ENV="${TRIPLE//-/_}"
TRIPLE_ENV="${TRIPLE_ENV^^}"
env "CARGO_TARGET_${TRIPLE_ENV}_LINKER=$HARF/bin/${TRIPLE}${API}-clang" \
    "CC=$HARF/bin/${TRIPLE}${API}-clang" \
    "AR=$HARF/bin/llvm-ar" \
  cargo build --release --target "$TRIPLE" -p lubot-arayuz
SO="$KOK/target/$TRIPLE/release/liblubot_arayuz.so"
test -f "$SO" || { echo "cdylib üretilmedi: $SO"; exit 1; }
cp "$SO" "$CIKTI/stage/lib/$ABI/"
echo "   $(du -h "$SO" | cut -f1)  liblubot_arayuz.so"

echo "== 3/8 korpus varlığı ve süzme =="
KORPUS="$KOK/corpus/knowledge-self.jsonl.gz"
test -f "$KORPUS" || { echo "korpus kurulmamış: $KORPUS (training/corpus_insa.py koş)"; exit 1; }
# Depodaki korpus arşivdir; pakete giden kopya **hizmet**tir. Süreç belgeleri
# (`served: false`) arşivde kalır, cihaza inmez: bkz. android/korpus_suz.py.
SUZULMUS="$CIKTI/suzulmus-korpus.jsonl.gz"
python3 "$KOK/android/korpus_suz.py" "$KORPUS" "$SUZULMUS" > "$CIKTI/korpus-suzme.json"
cat "$CIKTI/korpus-suzme.json"
cp "$SUZULMUS" "$CIKTI/stage/assets/korpus/knowledge-self.jsonl.gz"
echo "   $(du -h "$SUZULMUS" | cut -f1)  knowledge-self.jsonl.gz (süzülmüş)"

echo "== 4/8 kaynaklar (aapt2) =="
find "$KOK/android/res" -type f \( -name '*.xml' -o -name '*.png' \) -print0 \
  | xargs -0 -n1 aapt2 compile --dir "$KOK/android/res" -o "$CIKTI/obj/" 2>/dev/null \
  || aapt2 compile --dir "$KOK/android/res" -o "$CIKTI/obj/res.zip"
ls "$CIKTI/obj"

echo "== 5/8 bağla (aapt2 link) =="
aapt2 link \
  -o "$CIKTI/base.apk" \
  -I "$PLAT/android.jar" \
  --manifest "$KOK/android/AndroidManifest.xml" \
  --java "$CIKTI/gen" \
  --min-sdk-version "$API" --target-sdk-version 34 \
  $(find "$CIKTI/obj" -name '*.flat' -o -name 'res.zip')

echo "== 6/8 javac =="
mkdir -p "$CIKTI/obj/classes"
# Uyarı gürültüsü süzülüyor ama hata yutulmuyor: javac'ın çıkış kodu esas.
if ! javac -encoding UTF-8 -source 1.8 -target 1.8 -bootclasspath "$PLAT/android.jar" \
  -d "$CIKTI/obj/classes" \
  $(find "$CIKTI/gen" -name '*.java') \
  $(find "$KOK/android/src" -name '*.java') 2>"$CIKTI/javac.log"; then
  grep -v 'bootstrap class path\|source value 8\|target value 8' "$CIKTI/javac.log" || true
  echo "javac basarisiz"
  exit 1
fi
grep -v 'bootstrap class path\|source value 8\|target value 8\|deprecat' "$CIKTI/javac.log" || true

echo "== 7/8 dex (d8) =="
d8 --release --lib "$PLAT/android.jar" --output "$CIKTI/dex/" \
  $(find "$CIKTI/obj/classes" -name '*.class')
test -f "$CIKTI/dex/classes.dex"

echo "== 8/8 paketle, hizala, imzala =="
APK="$CIKTI/lubot-${ABI}-debug.apk"
cp "$CIKTI/base.apk" "$APK"
cp "$CIKTI/dex/classes.dex" "$CIKTI/stage/"
cd "$CIKTI/stage" && zip -qr "$APK" classes.dex lib assets && cd "$KOK"
zipalign -f -p 4 "$APK" "$CIKTI/hizali.apk"

KEYSTORE="$CIKTI/lubot-debug.keystore"
keytool -genkeypair -v -keystore "$KEYSTORE" -storepass lubotdebug -keypass lubotdebug \
  -alias lubot -keyalg RSA -keysize 2048 -validity 10000 \
  -dname "CN=Lubot, OU=lubot, O=budlum-xyz, L=Istanbul, C=TR" >/dev/null 2>&1

apksigner sign --ks "$KEYSTORE" --ks-pass pass:lubotdebug --key-pass pass:lubotdebug \
  --out "$CIKTI/lubot-$ABI.apk" "$CIKTI/hizali.apk"
apksigner verify --print-certs "$CIKTI/lubot-$ABI.apk" | head -4

# Paketin içinde olması gerekenler doğrulanıyor: eksik dex ya da eksik korpus
# ile imzalanmış bir APK "başarılı" sayılmaz.
for girdi in classes.dex "lib/$ABI/liblubot_arayuz.so" assets/korpus/knowledge-self.jsonl.gz AndroidManifest.xml resources.arsc; do
  unzip -l "$CIKTI/lubot-$ABI.apk" | grep -q "$girdi" || { echo "APK eksik: $girdi"; exit 1; }
done
echo "APK icerigi dogrulandi: dex + .so + korpus + manifest + arsc"

echo
echo "APK: $CIKTI/lubot-$ABI.apk  ($(du -h "$CIKTI/lubot-$ABI.apk" | cut -f1))"
