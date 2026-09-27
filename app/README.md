# app - the Android client, as a file

`lubot.apk` is built from this repository and is **not** checked in as source;
this directory holds the built artifact and the numbers behind it. The build
itself is `android/derle.sh`, and it needs two things the repository does not
carry: an Android SDK (for `aapt2`, `d8`, `zipalign`, `apksigner`) and an Android
NDK (for the aarch64 linker). Neither is vendored, and that is on purpose: a
600 MB toolchain in a source repository is not source.

```
ANDROID_SDK=~/android-sdk android/derle.sh
python3 android/apk_denetle.py android/out/lubot-arm64-v8a.apk --sdk ~/android-sdk
```

| what | value |
|---|---|
| package | `dev.budlum.lubot` |
| size | 9900878 bytes |
| min / target SDK | 24 / 34 |
| ABI | `arm64-v8a` only |
| permissions | `android.permission.INTERNET` (one, and checked to be one) |
| native | `lib/arm64-v8a/liblubot_arayuz.so`, 862888 bytes, 4 JNI functions |
| dex | `classes.dex`, 14792 bytes |
| corpus on device | `assets/korpus/knowledge-self.jsonl.gz`, 9498592 bytes |
| weights in the APK | none |
| signature | debug key (`CN=Lubot`), v1 block |

## Why the corpus is inside the APK

Earlier this file described an APK that carried no data and asked a node for
everything. That was wrong twice over: the numbers in it described a different
bridge (`crates/kopru`, three functions) than the one in the tree
(`crates/arayuz`, four functions, `kurulus` / `soru` / `belgeEkle` / `surum`),
and an app whose first screen is empty without a network is not the app this
repository is building. The corpus is 9.1 MB gzipped and 81673 records; it
rides along, and `derle.sh` refuses to sign an APK that is missing it. The
network is then used for what it is good for - asking a node a question - and
not for having something to show at all.

## What the client does and does not do

It opens the corpus shipped inside the package (`Kopru.kurulus`), masks secrets
in the question **on the device** before it goes anywhere, asks its own core
first (`Kopru.soru`), and prints what comes back. It can take a document the
user picks and add it to a separate record set with its own provenance
(`Kopru.belgeEkle`) - the device's own file never mixes into the corpus, so a
citation never points at the wrong origin.

It holds no threshold, no fallback text and no rule about what an answer may
say. The reply is either schema-checked Markdown or a `HATA:` line, and the
screen shows whichever arrived; it does not repair, rerank or soften either.

## What the build refuses to do

`derle.sh` fails rather than produce a package that looks fine:

* a missing tool is an error, not a skipped step;
* a missing corpus is an error (see above);
* `javac` is judged by its exit code, not by a filtered log - the filter only
  hides the bootstrap-classpath warnings Java 11 prints for `-source 8`;
* after signing, the package is opened again and must contain `classes.dex`,
  the `.so`, the corpus, the manifest and `resources.arsc`.

The last check exists because of a real mistake in this tree: the manifest once
said `xyz.budlum.lubot` while the Java source said `dev.budlum.lubot`, so
`aapt2` generated `R` under one name and the activity looked for it under
another. The build died with `package R does not exist`. The manifest and the
source now agree, and `gates/check.py::apk-sozlesmesi` reads both and fails if
they ever disagree again.
