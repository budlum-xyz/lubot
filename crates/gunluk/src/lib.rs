//! # `gunluk` - operator loglarini oku, ama once temizle
//!
//! Lubot bir operator katmanidir; operatorun elinde nginx erisim kayitlari ve
//! syslog vardir. Bu crate onlari **yapisal** hale getirir - satiri ayristirmak,
//! alanlari cikarmak - ve bunu yaparken tek bir kuraldan vazgecmez: okunan
//! satir once [`lubot_sir`]'den gecer.
//!
//! ## Neden once temizlik
//!
//! Loglar, sirlarin en sik sizdigi yerdir: bir istek satiri sorgu dizesinde
//! `?token=...` tasir, syslog'a bir komut satiri anahtariyla birlikte duser.
//! Ayristirmadan sonra temizlemek ise yarardi ama eksik olurdu: cikarilan alan
//! bir kayit defterine, oradan baska bir yere gider ve maskelenmemis bir alani
//! sonradan bulmak, hangi kopyalarin kirli oldugunu bilmeyi gerektirir. Bu
//! yuzden sirnak **girdide** calisir, cikarilan alanlarda degil.
//!
//! ## Bagimliligi yok
//!
//! Ayristirma elle yazilmistir: regex yok, tarih kutuphanesi yok. Sebep
//! kisitlama degil, davranistir - bir log satiri icin gereken sey alanlari
//! ayirmak ve basarisizligi **soylemek**, tarihi yorumlamak degil. Ayristirma
//! sinirli, deterministik ve satir satir calisir.
//!
//! ## Neyi iddia etmez
//!
//! Zaman damgasi bir dize olarak tasinir; bu crate onu cozmez ve siralamaz.
//! Bilinmeyen bir satir bicimi ayristirilamaz ve **reddedilir** - "kismen
//! ayristirilmis" bir kayit uretilmez, cunku yarim kayit, olmayan bir kayittan
//! daha tehlikelidir: varmis gibi gorunur.

/// Nginx erisim satirinin birlestirilmis biciminden cikarilan alanlar.
///
/// Alanlar metin olarak tasinir. Yuva adresi ve zaman damgasi birer etikettir;
/// bu crate onlari yorumlamaz, yalnizca dogru yerden cikarir.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NginxSatiri {
    /// Istegi yapan adres.
    pub adres: String,
    /// `[...]` icindeki zaman damgasi, oldugu gibi.
    pub zaman: String,
    /// `"..."` icindeki istek satiri.
    pub istek: String,
    /// Durum kodu.
    pub durum: u16,
    /// Gonderilen bayt sayisi.
    pub bayt: u64,
}

/// Nginx erisim satirini ayristirir.
///
/// Beklenen bicim:
/// `127.0.0.1 - - [14/Nov/2025:20:01:23 +0300] "GET /index.html HTTP/1.1" 200 1024`
///
/// # Errors
/// Bicim beklenenden farkliysa: adres, koseli parantez, tirnak, durum kodu ya
/// da bayt sayisi bulunamazsa. Hata mesaji **hangi alanin** eksik oldugunu
/// soyler; "gecersiz satir" demek, ayristiricinin isini kullaniciya yikmak olur.
pub fn nginx_satiri(satir: &str) -> Result<NginxSatiri, String> {
    let (adres, kalan) = satir
        .split_once(' ')
        .ok_or_else(|| "adres yok".to_string())?;
    let ac = kalan.find('[').ok_or_else(|| "zaman yok".to_string())?;
    let kapa_goreli = kalan[ac + 1..]
        .find(']')
        .ok_or_else(|| "zaman kapanmiyor".to_string())?;
    let kapa = ac + 1 + kapa_goreli;
    let zaman = kalan[ac + 1..kapa].to_string();
    let zamandan_sonra = &kalan[kapa + 1..];
    let t1 = zamandan_sonra
        .find('"')
        .ok_or_else(|| "istek yok".to_string())?;
    let t2_goreli = zamandan_sonra[t1 + 1..]
        .find('"')
        .ok_or_else(|| "istek kapanmiyor".to_string())?;
    let t2 = t1 + 1 + t2_goreli;
    let istek = zamandan_sonra[t1 + 1..t2].to_string();
    let kuyruk = zamandan_sonra[t2 + 1..].trim();
    let mut parcalar = kuyruk.split_whitespace();
    let durum = parcalar
        .next()
        .ok_or_else(|| "durum yok".to_string())?
        .parse::<u16>()
        .map_err(|e| format!("durum sayi degil: {e}"))?;
    let bayt = parcalar
        .next()
        .ok_or_else(|| "bayt yok".to_string())?
        .parse::<u64>()
        .map_err(|e| format!("bayt sayi degil: {e}"))?;
    Ok(NginxSatiri {
        adres: adres.to_string(),
        zaman,
        istek,
        durum,
        bayt,
    })
}

/// Syslog PRI degerinden `(facility, severity)`: ust uc bit ve alt uc bit.
#[must_use]
pub fn pri_coz(pri: u8) -> (u8, u8) {
    (pri >> 3, pri & 0x7)
}

/// Uygulama adindan facility cikarimi.
///
/// Yalniz yaygin ve **kesin** eslesmeler yapilir; taninmayan ad `None` doner.
/// `None` "bilinmiyor" demektir ve bilinmeyeni tahmin etmek, kaydi yanlis
/// sinifa koymak olurdu - orada kalici olarak kalirdi.
#[must_use]
pub fn facility_cikar(uygulama: Option<&str>) -> Option<u8> {
    let a = uygulama?.to_ascii_lowercase();
    if a.contains("sshd") || a.contains("sudo") || a.contains("pam") || a.contains("login") {
        Some(4)
    } else if a.contains("cron") {
        Some(9)
    } else {
        None
    }
}

/// Govdedeki yaygin kelimelere bakarak severity cikarimi.
///
/// Sirayla bakilir: en agir eslesme kazanir, cunku bir satir hem `info` hem
/// `failed` icerebilir ve o satir bir hatadir.
#[must_use]
pub fn severity_cikar(govde: &str) -> Option<u8> {
    let m = govde.to_ascii_lowercase();
    let agirlik: &[(&[&str], u8)] = &[
        (&["panic", "emerg"], 0),
        (&["alert"], 1),
        (&["crit"], 2),
        (&["fail", "failed", "error", "denied", "reddedildi"], 3),
        (&["warn", "warning", "uyari"], 4),
        (&["notice"], 5),
        (&["info", "started", "finished", "accepted"], 6),
        (&["debug"], 7),
    ];
    for (kelimeler, kod) in agirlik {
        if kelimeler.iter().any(|k| m.contains(k)) {
            return Some(*kod);
        }
    }
    None
}

/// Severity numarasinin adi. Sekizin ustundeki degerler `"unknown"` doner;
/// syslog PRI'si uc bittir ama cagiran taraf bir sayi uydurabilir.
#[must_use]
pub fn severity_adi(severity: u8) -> &'static str {
    match severity {
        0 => "emerg",
        1 => "alert",
        2 => "crit",
        3 => "err",
        4 => "warning",
        5 => "notice",
        6 => "info",
        7 => "debug",
        _ => "unknown",
    }
}

/// Ayristirilmis syslog olayi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyslogOlayi {
    /// PRI'den cozulen facility.
    pub facility: u8,
    /// PRI'den cozulen severity.
    pub severity: u8,
    /// Uygulama adi, varsa.
    pub uygulama: Option<String>,
    /// Govde.
    pub govde: String,
}

impl SyslogOlayi {
    /// Hem PRI'den hem govdeden gelen severity'yi birlestirir: **agir olan
    /// kazanir**.
    ///
    /// Bu, bir kayit defterinin en sik hatasini onler: bir satir `<14>` (info)
    /// ile isaretlenmis ama govdesi `failed` diyor. Ikisini de tasimak,
    /// cagirana hangisinin gercek oldugunu sormak olurdu; bu crate `failed`
    /// olani secer, cunku bir hatayi info diye kaydetmek, hatayi kaybetmektir.
    #[must_use]
    pub fn etkin_severity(&self) -> u8 {
        match severity_cikar(&self.govde) {
            Some(cikarilan) => cikarilan.min(self.severity),
            None => self.severity,
        }
    }
}

/// Syslog satirini ayristirir.
///
/// Giris bicimi: `<PRI>Oct 11 22:14:15 makine uygulama[pid]: govde`
///
/// # Errors
/// PRI oneki yoksa ya da sayi degilse. PRI, syslog'un bicimsel parcasi oldugu
/// icin yoklugu bir bicim hatasidir; ama **uygulama adi** yoklugu degildir ve
/// olay yine uretilir.
pub fn syslog_satiri(satir: &str) -> Result<SyslogOlayi, String> {
    let kirpik = satir.trim_start();
    let kapanis = kirpik
        .strip_prefix('<')
        .ok_or_else(|| "PRI oneki yok".to_string())?
        .find('>')
        .ok_or_else(|| "PRI kapanmiyor".to_string())?;
    let pri: u8 = kirpik[1..=kapanis]
        .parse()
        .map_err(|e| format!("PRI sayi degil: {e}"))?;
    let (facility, severity) = pri_coz(pri);
    // `kapanis` PRI'nin rakamlarindan sonraki `>` isaretinin **goreli**
    // indeksidir; mutlak konum bir fazlasi oldugu icin govde iki karakter
    // sonra baslar. Bir fazla alinan karakter, govdeyi `>` ile baslatir ve
    // uygulama adi taramasini bozar.
    let govde = kirpik[kapanis + 2..].trim();
    // Uygulama adi, `:` ile **biten** ilk sozcuktur. Bu kural, klasik
    // `Mmm dd hh:mm:ss makine uygulama[pid]: govde` bicimini oldugu kadar
    // zamansiz bicimi de kapsar ve iki tuzagi birden atlar: zaman damgasinin
    // icindeki iki nokta (`22:14:15`) bir sozcugu bitirmedigi icin uygulama
    // adi sanilmaz, makine adi da kendiliginden atlanir.
    let mut uygulama = None;
    let mut govde_basi = None;
    let mut konum = 0;
    for sozcuk in govde.split(' ') {
        if sozcuk.ends_with(':') && sozcuk.len() > 1 {
            let ad = sozcuk
                .trim_end_matches(':')
                .split_once('[')
                .map_or_else(|| sozcuk.trim_end_matches(':'), |(on, _)| on);
            uygulama = Some(ad.to_string());
            govde_basi = Some(konum + sozcuk.len());
            break;
        }
        konum += sozcuk.len() + 1;
    }
    let kalan = govde_basi.map_or_else(|| govde.to_string(), |b| govde[b..].trim().to_string());
    Ok(SyslogOlayi {
        facility,
        severity,
        uygulama,
        govde: kalan,
    })
}

/// Bir log satirini **temizleyip** ayristirir: once [`lubot_sir::maskele`],
/// sonra bicim.
///
/// Nginx satiri ise [`Satir::Nginx`], syslog ise [`Satir::Syslog`] doner;
/// ikisi de degilse `None`. Siralama bilinclidir: syslog PRI oneki tasir,
/// nginx tasimaz, ve bir satiri iki bicimden birine zorlamak yerine
/// taninmayan satir disarida kalir.
#[must_use]
pub fn satir_oku(satir: &str) -> Option<Satir> {
    let temiz = lubot_sir::maskele(satir);
    let metin = temiz.metin();
    if metin.trim_start().starts_with('<') {
        return syslog_satiri(metin).ok().map(Satir::Syslog);
    }
    nginx_satiri(metin).ok().map(Satir::Nginx)
}

/// Bir satirin ayristirilmis hali.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Satir {
    /// Nginx erisim satiri.
    Nginx(NginxSatiri),
    /// Syslog olayi.
    Syslog(SyslogOlayi),
}

/// Log govdesini satir satir okur ve yalniz ayristirilabilenleri dondurur.
///
/// Atilan satir sayisi da doner: "kac satir okundu" tek basina bir olcum
/// degildir, "kac satir **anlasilmadi**" olcumdur. Bir ayristirici sessizce
/// satir atliyorsa, eksik veriyi tam saniyorsun demektir.
#[must_use]
pub fn oku(govde: &str) -> (Vec<Satir>, usize) {
    let mut satirlar = Vec::new();
    let mut atilan = 0;
    for satir in govde.lines() {
        if satir.trim().is_empty() {
            continue;
        }
        match satir_oku(satir) {
            Some(s) => satirlar.push(s),
            None => atilan += 1,
        }
    }
    (satirlar, atilan)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NGINX: &str =
        "127.0.0.1 - - [14/Nov/2025:20:01:23 +0300] \"GET /index.html HTTP/1.1\" 200 1024";

    #[test]
    fn nginx_satiri_alanlara_ayrilir() {
        let s = nginx_satiri(NGINX).expect("ayristirilamadi");
        assert_eq!(s.adres, "127.0.0.1");
        assert_eq!(s.zaman, "14/Nov/2025:20:01:23 +0300");
        assert_eq!(s.istek, "GET /index.html HTTP/1.1");
        assert_eq!(s.durum, 200);
        assert_eq!(s.bayt, 1024);
    }

    #[test]
    fn eksik_alan_adi_soylenir() {
        assert!(nginx_satiri("127.0.0.1 - - [x] \"GET /\" 200")
            .expect_err("bayt olmadan gecti")
            .contains("bayt"));
        assert!(nginx_satiri("127.0.0.1 tek basina")
            .expect_err("zaman olmadan gecti")
            .contains("zaman"));
    }

    #[test]
    fn durum_sayi_degilse_reddedilir() {
        let hata = nginx_satiri("1.2.3.4 - - [z] \"GET /\" ABC 12").expect_err("gecti");
        assert!(hata.contains("durum sayi degil"));
    }

    #[test]
    fn pri_ust_ve_alt_uc_bite_ayrilir() {
        assert_eq!(pri_coz(34), (4, 2));
        assert_eq!(pri_coz(0), (0, 0));
        assert_eq!(pri_coz(191), (23, 7));
    }

    #[test]
    fn severity_adi_yedi_seviyeyi_tanir() {
        let adlar: Vec<&str> = (0..=7).map(severity_adi).collect();
        assert_eq!(
            adlar,
            vec!["emerg", "alert", "crit", "err", "warning", "notice", "info", "debug"]
        );
        assert_eq!(severity_adi(9), "unknown");
    }

    #[test]
    fn severity_cikarimi_en_agiri_secer() {
        assert_eq!(severity_cikar("auth failed for root"), Some(3));
        assert_eq!(severity_cikar("service started ok"), Some(6));
        assert_eq!(severity_cikar("kernel panic"), Some(0));
        assert_eq!(severity_cikar("sadece bir cumle"), None);
    }

    #[test]
    fn facility_yalniz_kesin_eslesmede_doner() {
        assert_eq!(facility_cikar(Some("sshd")), Some(4));
        assert_eq!(facility_cikar(Some("cron")), Some(9));
        assert_eq!(facility_cikar(Some("bilinmeyen-uygulama")), None);
        assert_eq!(facility_cikar(None), None);
    }

    #[test]
    fn syslog_satiri_ayristirilir() {
        let olay =
            syslog_satiri("<34>Oct 11 22:14:15 makine su[123]: 'su root' failed").expect("olmadi");
        assert_eq!(olay.facility, 4);
        assert_eq!(olay.severity, 2);
        assert_eq!(olay.uygulama.as_deref(), Some("su"));
        assert_eq!(olay.govde, "'su root' failed");
    }

    #[test]
    fn uygulama_adi_yoksa_olay_yine_uretilir() {
        let olay = syslog_satiri("<13>bir govde iki noktasi olmadan").expect("olmadi");
        assert_eq!(olay.uygulama, None);
        assert!(olay.govde.contains("iki noktasi"));
    }

    #[test]
    fn pri_yoksa_reddedilir() {
        assert!(syslog_satiri("Oct 11 22:14:15 makine su: x").is_err());
        assert!(syslog_satiri("<abc>x").is_err());
    }

    #[test]
    fn etkin_severity_govdedeki_agiri_secer() {
        // PRI info diyor, govde hata: kayit hata olmali.
        let olay = syslog_satiri("<14>Oct 11 22:14:15 m app: connection failed").expect("olmadi");
        assert_eq!(olay.severity, 6);
        assert_eq!(olay.etkin_severity(), 3);
    }

    #[test]
    fn etkin_severity_govde_susuyorsa_pri_kalir() {
        let olay = syslog_satiri("<11>Oct 11 22:14:15 m app: something").expect("olmadi");
        assert_eq!(olay.etkin_severity(), 3);
    }

    #[test]
    fn okuma_once_sir_maskeler() {
        let sir = format!("ghp_{}", "abcdefghijklmnopqrstuvwxyz123456");
        let govde = format!("1.2.3.4 - - [z] \"GET /?token={sir} HTTP/1.1\" 200 12");
        let (satirlar, atilan) = oku(&govde);
        assert_eq!(atilan, 0);
        assert_eq!(satirlar.len(), 1);
        match &satirlar[0] {
            Satir::Nginx(s) => {
                assert!(s.istek.contains(lubot_sir::MASKE));
                assert!(!s.istek.contains("ghp_"));
            }
            Satir::Syslog(_) => panic!("nginx satiri syslog sanildi"),
        }
    }

    #[test]
    fn satir_oku_iki_bicimi_de_tanir_ve_kalanini_reddeder() {
        assert!(matches!(satir_oku(NGINX), Some(Satir::Nginx(_))));
        assert!(matches!(
            satir_oku("<13>Oct 11 22:14:15 m app: x"),
            Some(Satir::Syslog(_))
        ));
        assert!(satir_oku("bu bir log satiri degil").is_none());
    }

    #[test]
    fn atilan_satir_sayisi_bildirilir() {
        let govde = format!("{NGINX}\nsadece duz metin\n<13>Oct 11 22:14:15 m app: ok\n");
        let (satirlar, atilan) = oku(&govde);
        assert_eq!(satirlar.len(), 2);
        assert_eq!(atilan, 1);
    }

    #[test]
    fn bos_satirlar_atilan_sayilmaz() {
        let (satirlar, atilan) = oku("\n\n  \n");
        assert!(satirlar.is_empty());
        assert_eq!(atilan, 0);
    }
}
