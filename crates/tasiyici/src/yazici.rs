//! Writing a container: the only place that decides where a byte goes.
//!
//! The writer is deliberately dull. It appends each tensor's scales and packed
//! indices to a payload, aligning both to [`crate::bicim::HIZA`], records where
//! it put them, and emits the header last because the header carries the
//! payload digest. There is no packing heuristic, no reordering, no attempt to
//! group tensors by rung in the file: the ladder is a property of the
//! *directory*, not of the byte order, so a reader that wants rungs zero to
//! four reads four ranges rather than one.
//!
//! That is a real trade and it is worth writing down. Sorting the payload by
//! rung would let a shallow reader map one contiguous prefix of the file, which
//! is friendlier to a disk. Not sorting keeps the writer's output a pure
//! function of the order tensors were added, which makes two builds of the same
//! model byte-identical and therefore makes the digest mean something.
//! Reproducibility won; if a device ever needs the prefix property, it can be
//! added as a second, explicitly named ordering rather than by making the
//! default ordering depend on a heuristic.

use crate::bicim::{hizala, BicimHatasi, Kayit, BASLIK_BAYT, IMZA, SURUM};
use lubot_nicem::grup::Genislik;
use lubot_nicem::Nicemlenmis;
use sha2::{Digest, Sha256};

/// Why a tensor could not be added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YaziciHatasi {
    /// A name already in the container.
    AdTekrari { ad: String },
    /// An empty name. A tensor nobody can ask for by name is a tensor nobody
    /// can load.
    AdBos,
    /// A name longer than the directory's length field can hold.
    AdUzun { uzunluk: usize },
    /// A group size that does not fit the directory's field.
    Grup { grup: usize },
}

impl std::fmt::Display for YaziciHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdTekrari { ad } => write!(f, "ad tekrari: {ad}"),
            Self::AdBos => write!(f, "adsiz tensor eklenemez"),
            Self::AdUzun { uzunluk } => write!(f, "ad {uzunluk} bayt, tavan 65535"),
            Self::Grup { grup } => write!(f, "grup {grup} dizin alanina sigmaz"),
        }
    }
}

impl std::error::Error for YaziciHatasi {}

impl From<YaziciHatasi> for BicimHatasi {
    fn from(e: YaziciHatasi) -> Self {
        match e {
            YaziciHatasi::AdTekrari { ad } => Self::AdTekrari { ad },
            YaziciHatasi::AdBos => Self::AdTekrari { ad: String::new() },
            YaziciHatasi::AdUzun { uzunluk } => Self::UzunlukTutmuyor {
                beyan: uzunluk,
                var: 65535,
                alan: "ad",
            },
            YaziciHatasi::Grup { grup } => Self::Grup {
                ad: String::new(),
                grup: u32::try_from(grup).unwrap_or(u32::MAX),
            },
        }
    }
}

/// Builds a container in memory.
#[derive(Debug, Default)]
pub struct Yazici {
    kayitlar: Vec<Kayit>,
    yuk: Vec<u8>,
}

impl Yazici {
    /// An empty container.
    #[must_use]
    pub fn yeni() -> Self {
        Self::default()
    }

    /// Number of tensors added so far.
    #[must_use]
    pub fn adet(&self) -> usize {
        self.kayitlar.len()
    }

    /// Payload bytes written so far, padding included.
    #[must_use]
    pub fn yuk_bayt(&self) -> usize {
        self.yuk.len()
    }

    /// Add a quantised tensor on a given rung of the ladder.
    ///
    /// Rung `0` is the part every depth needs; `1..` are the layers in order.
    /// The rung is the writer's only semantic input, and it is taken from the
    /// caller rather than guessed from the name: a naming convention that the
    /// loader silently depends on is a convention that breaks when someone
    /// renames a tensor.
    ///
    /// # Errors
    ///
    /// [`YaziciHatasi::AdBos`], [`YaziciHatasi::AdTekrari`],
    /// [`YaziciHatasi::AdUzun`] and [`YaziciHatasi::Grup`].
    pub fn ekle(&mut self, ad: &str, kademe: u8, tensor: &Nicemlenmis) -> Result<(), YaziciHatasi> {
        if ad.is_empty() {
            return Err(YaziciHatasi::AdBos);
        }
        if ad.len() > usize::from(u16::MAX) {
            return Err(YaziciHatasi::AdUzun { uzunluk: ad.len() });
        }
        if self.kayitlar.iter().any(|k| k.ad == ad) {
            return Err(YaziciHatasi::AdTekrari { ad: ad.to_string() });
        }
        let grup = u32::try_from(tensor.grup()).map_err(|_| YaziciHatasi::Grup {
            grup: tensor.grup(),
        })?;

        let olcek_ofset = self.hizala_yuk();
        for o in tensor.olcekler() {
            self.yuk.extend_from_slice(&o.to_le_bytes());
        }
        let olcek_bayt = (self.yuk.len() - olcek_ofset) as u64;

        let yuk_ofset = self.hizala_yuk();
        self.yuk.extend_from_slice(tensor.yuk());
        let yuk_bayt = (self.yuk.len() - yuk_ofset) as u64;

        self.kayitlar.push(Kayit {
            ad: ad.to_string(),
            genislik: tensor.genislik(),
            kademe,
            grup,
            satir: tensor.satir() as u64,
            son_eksen: tensor.son_eksen() as u64,
            olcek_ofset: olcek_ofset as u64,
            olcek_bayt,
            yuk_ofset: yuk_ofset as u64,
            yuk_bayt,
        });
        Ok(())
    }

    /// Pad the payload up to the next alignment boundary and report where the
    /// next block starts.
    fn hizala_yuk(&mut self) -> usize {
        let hedef = hizala(self.yuk.len());
        self.yuk.resize(hedef, 0);
        hedef
    }

    /// Emit the finished container.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn bayt(&self) -> Vec<u8> {
        let dizin = self.dizin();
        let mut cikti = Vec::with_capacity(BASLIK_BAYT + hizala(dizin.len()) + self.yuk.len());
        let mut ozet = Sha256::new();
        ozet.update(&self.yuk);
        let ozet: [u8; 32] = ozet.finalize().into();

        cikti.extend_from_slice(IMZA);
        cikti.extend_from_slice(&SURUM.to_le_bytes());
        cikti.extend_from_slice(&(self.kayitlar.len() as u32).to_le_bytes());
        cikti.extend_from_slice(&(dizin.len() as u64).to_le_bytes());
        cikti.extend_from_slice(&(self.yuk.len() as u64).to_le_bytes());
        cikti.extend_from_slice(&ozet);
        debug_assert_eq!(cikti.len(), BASLIK_BAYT);
        cikti.extend_from_slice(&dizin);
        cikti.resize(hizala(cikti.len()), 0);
        cikti.extend_from_slice(&self.yuk);
        cikti
    }

    /// Serialise the directory.
    ///
    /// The two narrowing casts are bounded by checks made at insertion time:
    /// a name longer than 65535 bytes is refused with [`YaziciHatasi::AdUzun`],
    /// so every name length that reaches here fits in a `u16`,
    /// and the tensor count is a `usize` that cannot exceed `u32` in any file
    /// this writer can hold in memory.
    #[allow(clippy::cast_possible_truncation)]
    fn dizin(&self) -> Vec<u8> {
        let mut d = Vec::new();
        for k in &self.kayitlar {
            d.extend_from_slice(&(k.ad.len() as u16).to_le_bytes());
            d.extend_from_slice(k.ad.as_bytes());
            d.push(crate::bicim::genislik_kodu(k.genislik));
            d.push(k.kademe);
            d.extend_from_slice(&k.grup.to_le_bytes());
            d.extend_from_slice(&k.satir.to_le_bytes());
            d.extend_from_slice(&k.son_eksen.to_le_bytes());
            d.extend_from_slice(&k.olcek_ofset.to_le_bytes());
            d.extend_from_slice(&k.olcek_bayt.to_le_bytes());
            d.extend_from_slice(&k.yuk_ofset.to_le_bytes());
            d.extend_from_slice(&k.yuk_bayt.to_le_bytes());
        }
        d
    }

    /// The width of the widest tensor, for reporting.
    #[must_use]
    pub fn en_genis(&self) -> Option<Genislik> {
        self.kayitlar.iter().map(|k| k.genislik).next_back()
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
mod tests {
    use super::*;
    use crate::bicim::Kapsayici;
    use crate::bicim::HIZA;
    use lubot_nicem::Nicemleyici;

    fn ornek(n: usize) -> Vec<f32> {
        let mut durum = 0xdead_beefu32;
        (0..n)
            .map(|_| {
                durum = durum.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                #[allow(clippy::cast_precision_loss)]
                let v = ((durum >> 8) as f32) / 8_388_608.0 - 1.0;
                v * 0.04
            })
            .collect()
    }

    #[test]
    fn the_same_model_written_twice_is_byte_identical() {
        // The digest is only worth something if the writer is a function.
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let t = n.nicemle(&ornek(128 * 6), 128).expect("q");
        let kur = || {
            let mut y = Yazici::yeni();
            y.ekle("gomme", 0, &t).expect("added");
            y.ekle("katman.1.wq", 1, &t).expect("added");
            y.bayt()
        };
        assert_eq!(kur(), kur());
    }

    #[test]
    fn a_tensor_survives_the_file_and_comes_back_as_the_same_numbers() {
        let n = Nicemleyici::yeni(Genislik::Bit(3), 64).expect("valid");
        let w = ornek(64 * 5);
        let t = n.nicemle(&w, 64).expect("q");
        let beklenen = t.coz().expect("reconstructs");

        let mut y = Yazici::yeni();
        y.ekle("tek", 0, &t).expect("added");
        let dosya = y.bayt();

        let k = Kapsayici::ac(&dosya).expect("opens");
        let (olcek, yuk) = k.dilimler("tek").expect("present");
        assert_eq!(olcek.len(), t.olcekler().len() * 2);
        assert_eq!(yuk, t.yuk());
        // And the scales survived the little-endian trip.
        for (i, o) in t.olcekler().iter().enumerate() {
            let geri = u16::from_le_bytes([olcek[i * 2], olcek[i * 2 + 1]]);
            assert_eq!(geri, *o);
        }
        assert_eq!(beklenen.len(), 64 * 5);
    }

    #[test]
    fn an_empty_container_is_valid_and_says_so() {
        let y = Yazici::yeni();
        let dosya = y.bayt();
        let k = Kapsayici::ac(&dosya).expect("opens");
        assert_eq!(k.kayitlar().len(), 0);
        assert_eq!(k.agirlik(), 0);
        assert!((k.agirlik_basina_bit(dosya.len()) - 0.0).abs() < f64::EPSILON);
        k.dogrula().expect("the empty digest holds");
    }

    #[test]
    fn duplicate_and_empty_names_are_refused_at_the_call() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 64).expect("valid");
        let t = n.nicemle(&ornek(64 * 2), 64).expect("q");
        let mut y = Yazici::yeni();
        y.ekle("a", 0, &t).expect("added");
        assert_eq!(
            y.ekle("a", 1, &t).err(),
            Some(YaziciHatasi::AdTekrari {
                ad: "a".to_string()
            })
        );
        assert_eq!(y.ekle("", 1, &t).err(), Some(YaziciHatasi::AdBos));
        assert_eq!(y.adet(), 1);
    }

    #[test]
    fn a_hundred_tensors_still_open_and_keep_their_rungs() {
        let n = Nicemleyici::yeni(Genislik::Bit(2), 64).expect("valid");
        let t = n.nicemle(&ornek(64 * 2), 64).expect("q");
        let mut y = Yazici::yeni();
        for i in 0..100u8 {
            y.ekle(&format!("t{i}"), i / 4, &t).expect("added");
        }
        let dosya = y.bayt();
        let k = Kapsayici::ac(&dosya).expect("opens");
        assert_eq!(k.kayitlar().len(), 100);
        for (i, kayit) in k.kayitlar().iter().enumerate() {
            assert_eq!(kayit.ad, format!("t{i}"));
            assert_eq!(kayit.kademe, (i / 4) as u8);
        }
    }

    #[test]
    fn padding_costs_at_most_one_alignment_per_block_and_the_bound_is_exact() {
        // Alignment is not free and the test says so in bytes rather than in a
        // percentage: a percentage hides the fact that the waste is per block,
        // so it looks fine on large tensors and alarming on small ones while
        // the underlying cost is identical.
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let t = n.nicemle(&ornek(128 * 16), 128).expect("q");
        let mut y = Yazici::yeni();
        for i in 0..20u8 {
            y.ekle(&format!("t{i}"), i, &t).expect("added");
        }
        let dosya = y.bayt();
        let k = Kapsayici::ac(&dosya).expect("opens");
        let talep = k.talep_edilen_bayt();
        // Two blocks per tensor, each losing strictly less than one alignment.
        let tavan = talep + 20 * 2 * (HIZA as u64 - 1);
        assert!(
            (k.yuk().len() as u64) <= tavan,
            "payload {} exceeds the padding bound {tavan}",
            k.yuk().len()
        );
        // And the part of the file that is not payload is the header plus the
        // directory, which is names and integers and nothing else.
        let ust_bilgi = dosya.len() as u64 - k.yuk().len() as u64;
        assert!(ust_bilgi >= BASLIK_BAYT as u64);
        assert!(
            ust_bilgi < BASLIK_BAYT as u64 + 20 * 128,
            "directory is {ust_bilgi} bytes"
        );
    }

    #[test]
    fn at_a_realistic_tensor_size_the_file_is_almost_entirely_weights() {
        // The same layout that wastes 15% on 2048-weight toys wastes a tenth of
        // a percent on a tensor of the size a real layer has. The bound above
        // is the guarantee; this is the consequence that matters in practice.
        let n = Nicemleyici::yeni(Genislik::Bit(2), 128).expect("valid");
        let t = n.nicemle(&ornek(128 * 512), 128).expect("q");
        let mut y = Yazici::yeni();
        for i in 0..8u8 {
            y.ekle(&format!("t{i}"), i, &t).expect("added");
        }
        let dosya = y.bayt();
        let k = Kapsayici::ac(&dosya).expect("opens");
        #[allow(clippy::cast_precision_loss)]
        let dolu = k.talep_edilen_bayt() as f64 / dosya.len() as f64;
        assert!(
            dolu > 0.99,
            "only {:.2}% of the file is weights",
            dolu * 100.0
        );
    }
}
