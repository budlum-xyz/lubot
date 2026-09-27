package dev.budlum.lubot;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Lubot'un tek ekranı — sohbet akışı, koyu tema.
 *
 * Arayüz bilerek ince: ne bir model çağırır ne bir şey üretir. Yaptığı üç iş
 * var — korpusu cihaza açmak, soruyu Rust çekirdeğine götürmek, cihazdan
 * seçilen belgeyi izole kayda eklemek. Cevabın doğrulanması arayüzde değil,
 * çekirdekte olur; burası yalnızca gösterir. AndroidX yok: balonlar
 * GradientDrawable ile programatik çiziliyor.
 */
public class AnaEtkinlik extends Activity {

    private static final int BELGE_SEC = 4711;
    private static final String KORPUS_VARLIGI = "korpus/knowledge-self.jsonl.gz";

    private static final int RENK_BALON_KULLANICI = Color.parseColor("#1F6FEB");
    private static final int RENK_BALON_LUBOT = Color.parseColor("#21262D");
    private static final int RENK_YAZI = Color.parseColor("#E6EDF3");
    private static final int RENK_HATA = Color.parseColor("#F85149");

    private TextView durum;
    private LinearLayout akis;
    private ScrollView kaydirma;
    private EditText soruKutusu;
    private volatile boolean hazir = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.ana);

        durum = (TextView) findViewById(R.id.durum);
        akis = (LinearLayout) findViewById(R.id.akis);
        kaydirma = (ScrollView) findViewById(R.id.kaydirma);
        soruKutusu = (EditText) findViewById(R.id.soru);
        Button sorDugmesi = (Button) findViewById(R.id.dugme_sor);
        Button belgeDugmesi = (Button) findViewById(R.id.dugme_belge);

        sorDugmesi.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { soruSor(); }
        });
        belgeDugmesi.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View v) { belgeSec(); }
        });

        balonEkle(getString(R.string.bos_cevap), false);
        kurulusuBaslat();
    }

    /** Karar verici dp dönüşümü: pikseli burada üretilir, layout içinde kullanılır. */
    private int dp(int deger) {
        return (int) (deger * getResources().getDisplayMetrics().density + 0.5f);
    }

    /**
     * Akışa bir balon ekler. Kullanıcı balonu sağda ve mavi; Lubot balonu
     * solda ve yuzey grisi. HATA öneki taşıyan metin kırmızıya boyanır —
     * hata yutulmuyor, olduğu gibi gösteriliyor.
     */
    private void balonEkle(String metin, boolean kullanici) {
        TextView balon = new TextView(this);
        balon.setText(metin);
        balon.setTextSize(14f);
        balon.setTextIsSelectable(true);
        balon.setTextColor(metin.startsWith("HATA:") ? RENK_HATA : RENK_YAZI);
        int yatay = dp(12);
        int dikey = dp(9);
        balon.setPadding(yatay, dikey, yatay, dikey);

        GradientDrawable zemin = new GradientDrawable();
        zemin.setColor(kullanici ? RENK_BALON_KULLANICI : RENK_BALON_LUBOT);
        zemin.setCornerRadius(dp(16));
        balon.setBackground(zemin);

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.gravity = kullanici ? Gravity.END : Gravity.START;
        int boslukYan = dp(56);
        int boslukUst = dp(6);
        if (kullanici) {
            lp.setMargins(boslukYan, boslukUst, 0, 0);
        } else {
            lp.setMargins(0, boslukUst, boslukYan, 0);
        }
        balon.setLayoutParams(lp);

        akis.addView(balon);
        kaydirma.post(new Runnable() {
            @Override public void run() {
                kaydirma.fullScroll(View.FOCUS_DOWN);
            }
        });
    }

    /** Özet JSON'unu insanın okuyacağı tek satıra indirir. */
    private String hazirSatiri(String ozet) {
        String kayit = alan(ozet, "kayit");
        String jeton = alan(ozet, "toplam_jeton");
        String surum = Kopru.surum();
        return "Hazır · " + kayit + " kayıt · " + jeton + " jeton · " + surum;
    }

    /** JSON'dan tek bir sayı alanını çeker; yoksa "?" döner, uydurmaz. */
    private static String alan(String json, String ad) {
        int i = json.indexOf("\"" + ad + "\"");
        if (i < 0) return "?";
        int ikiNokta = json.indexOf(':', i);
        if (ikiNokta < 0) return "?";
        int bas = ikiNokta + 1;
        int son = bas;
        while (son < json.length()
                && (Character.isDigit(json.charAt(son)) || json.charAt(son) == ' ')) son++;
        return json.substring(bas, son).trim();
    }

    private void soruSor() {
        if (!hazir) {
            Toast.makeText(this, "Korpus hazır değil", Toast.LENGTH_SHORT).show();
            return;
        }
        final String soru = soruKutusu.getText().toString().trim();
        if (soru.isEmpty()) {
            Toast.makeText(this, "Soru boş", Toast.LENGTH_SHORT).show();
            return;
        }
        balonEkle(soru, true);
        soruKutusu.setText("");
        durum.setText("Okunuyor…");
        final Activity self = this;
        yeniIsParcacigi(new Runnable() {
            @Override public void run() {
                final String sonuc = Kopru.soru(soru);
                self.runOnUiThread(new Runnable() {
                    @Override public void run() {
                        balonEkle(sonuc, false);
                        durum.setText(sonuc.startsWith("HATA:")
                                ? sonuc
                                : "Cevap alındı · okuyucu: android-arayuz");
                    }
                });
            }
        });
    }

    private void belgeSec() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("text/*");
        try {
            startActivityForResult(i, BELGE_SEC);
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.belge_okunamadi) + ": " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onActivityResult(int istekKodu, int sonucKodu, Intent veri) {
        super.onActivityResult(istekKodu, sonucKodu, veri);
        if (istekKodu != BELGE_SEC || sonucKodu != RESULT_OK || veri == null) return;
        final Uri uri = veri.getData();
        if (uri == null) return;
        durum.setText("Belge okunuyor…");
        final Activity self = this;
        yeniIsParcacigi(new Runnable() {
            @Override public void run() {
                String gecici;
                try {
                    String icerik = uriOku(uri);
                    gecici = Kopru.belgeEkle(adBul(uri), icerik,
                            getFilesDir().getAbsolutePath());
                } catch (Exception e) {
                    gecici = "HATA: " + e.getClass().getSimpleName() + ": " + e.getMessage();
                }
                final String sonuc = gecici;
                self.runOnUiThread(new Runnable() {
                    @Override public void run() {
                        if (sonuc.startsWith("HATA:")) {
                            durum.setText(sonuc);
                            balonEkle(sonuc, false);
                        } else {
                            durum.setText("Belge eklendi (izole kayıt, korpus dışı)");
                            balonEkle("Belge alındı — izole kayıtta duruyor, korpusa karışmadı. "
                                    + "Kayıt: " + sonuc.substring(0, Math.min(12, sonuc.length())), false);
                        }
                    }
                });
            }
        });
    }

    private String uriOku(Uri uri) throws Exception {
        InputStream girdi = getContentResolver().openInputStream(uri);
        if (girdi == null) throw new IllegalStateException("akış açılamadı");
        try {
            ByteArrayOutputStream tampon = new ByteArrayOutputStream();
            byte[] parca = new byte[8192];
            int n;
            while ((n = girdi.read(parca)) > 0) tampon.write(parca, 0, n);
            return new String(tampon.toByteArray(), StandardCharsets.UTF_8);
        } finally {
            girdi.close();
        }
    }

    private String adBul(Uri uri) {
        String yol = uri.getLastPathSegment();
        if (yol == null || yol.isEmpty()) return "cihaz-belgesi.md";
        int kesme = yol.lastIndexOf('/');
        return kesme >= 0 ? yol.substring(kesme + 1) : yol;
    }

    /** Korpusu varlıklardan cihaza çıkarır ve köprüyü arka planda kurar. */
    private void kurulusuBaslat() {
        final Activity self = this;
        yeniIsParcacigi(new Runnable() {
            @Override public void run() {
                String sonuc;
                try {
                    File korpusDizini = varligiCikar();
                    sonuc = Kopru.kurulus(korpusDizini.getAbsolutePath(),
                            getFilesDir().getAbsolutePath());
                } catch (Exception e) {
                    sonuc = "HATA: " + e.getClass().getSimpleName() + ": " + e.getMessage();
                }
                final String s = sonuc;
                self.runOnUiThread(new Runnable() {
                    @Override public void run() {
                        if (s.startsWith("HATA:")) {
                            durum.setText(s);
                            hazir = false;
                        } else {
                            durum.setText(hazirSatiri(s));
                            hazir = true;
                        }
                    }
                });
            }
        });
    }

    /** Korpusu APK varlıklarından uygulama dizinine bir kez çıkarır. */
    private File varligiCikar() throws Exception {
        File dizin = new File(getFilesDir(), "korpus");
        File hedef = new File(dizin, "knowledge-self.jsonl.gz");
        if (hedef.isFile() && hedef.length() > 0) return dizin;
        if (!dizin.isDirectory() && !dizin.mkdirs()) {
            throw new IllegalStateException("korpus dizini oluşturulamadı");
        }
        InputStream girdi = getAssets().open(KORPUS_VARLIGI);
        try {
            OutputStream cikti = new FileOutputStream(hedef);
            try {
                byte[] tampon = new byte[16384];
                int n;
                while ((n = girdi.read(tampon)) > 0) cikti.write(tampon, 0, n);
            } finally {
                cikti.close();
            }
        } finally {
            girdi.close();
        }
        return dizin;
    }

    private void yeniIsParcacigi(Runnable is) {
        new Thread(is).start();
    }
}
