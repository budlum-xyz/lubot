package dev.budlum.lubot;

/**
 * Lubot'un Rust çekirdeğine açılan kapı.
 *
 * Burada hiçbir mantık yok: her metot doğrudan {@code crates/arayuz}'deki JNI
 * dışa aktarımına gider ve aynı {@code ask} yolunu koşar. Dönen dize ya
 * şema-doğrulanmış Markdown'dır ya da {@code HATA:} ile başlayan bir red;
 * ikisi aynı satırda durmaz ve arayüz ikisini olduğu gibi gösterir.
 */
public final class Kopru {

    static {
        System.loadLibrary("lubot_arayuz");
    }

    /** Köprü sürümü; JNI bağının gerçekten kurulduğunu da doğrular. */
    public static native String surum();

    /**
     * Korpusu yükler, izin defterini sıfırlar.
     *
     * @param korpusDizini  içinde .jsonl ya da .jsonl.gz dosyaları olan dizin
     * @param calismaDizini denetim kaydı ve cihaz belgelerinin yazılacağı dizin
     * @return korpus özeti (JSON) ya da "HATA: ..."
     */
    public static native String kurulus(String korpusDizini, String calismaDizini);

    /** Soruyu sorar. Şema-doğrulanmış Markdown ya da "HATA: ..." döner. */
    public static native String soru(String soru);

    /**
     * Cihazdan gelen bir metni izole kayda ekler ve korpusu yeniden yükler.
     *
     * Kayıt korpusa karışmaz: kendi kaynağı ({@code cihaz}) ve lisansıyla
     * ayrı dosyada durur, böylece alıntının nereden geldiği karışmaz.
     *
     * @return kayıt özeti (content_id) ya da "HATA: ..."
     */
    public static native String belgeEkle(String ad, String icerik, String calismaDizini);

    private Kopru() {}
}
