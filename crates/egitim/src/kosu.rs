//! The training run: a schedule, an epoch rule, and a report that says why it
//! stopped.
//!
//! # What this module is responsible for
//!
//! Not the arithmetic - that is the trainer's job - but the *discipline* around
//! it. Three things go wrong quietly in a from-scratch run, and each is a rule
//! here rather than a comment:
//!
//! 1. **A resumed run is not the run it continues.** The optimizer's step count
//!    enters the bias correction, the epoch counter enters the window shuffle,
//!    and the learning-rate schedule is a function of the step. A resumed run
//!    that starts all three at zero trains a different model and reports it
//!    under the same name. So the caller hands in where the run *is*
//!    ([`KosuAyari::baslangic_adim`], [`KosuAyari::baslangic_epoch`],
//!    [`KosuAyari::baslangic_en_iyi`]) and the loop picks up from there.
//! 2. **A loss that stops falling is not progress.** The epoch rule stops the
//!    run when an epoch's validation loss fails to beat the best one carried
//!    in, and the reason is recorded as [`DurmaNedeni::EpochKurali`] rather
//!    than being rounded into "the step budget ran out".
//! 3. **A short run must not be judged by its own average.** The learning rate
//!    schedule is a function of the *planned* horizon, not of the steps this
//!    call happens to run, so two halves of a run produce the same curve as the
//!    whole. Without [`KosuAyari::planlanan_adim`] the second half would see a
//!    schedule that never reached its floor.

use crate::veri::{Bolum, Kayit};
use crate::{ileri_ve_geri_paket, paketle, Adamw, Parametreler, Spec};

/// The schedule, the budget and where the run is picking up.
#[derive(Debug, Clone, PartialEq)]
pub struct KosuAyari {
    /// The architecture being trained.
    pub spec: Spec,
    /// Window length, in tokens.
    pub pencere_uzunlugu: usize,
    /// Seed for the shuffle and (elsewhere) for the initialisation.
    pub tohum: u64,
    /// Peak learning rate.
    pub ogrenme_orani: f64,
    /// Decoupled weight decay, applied through the parameter mask.
    pub agirlik_sonumu: f64,
    /// Linear warm-up, in steps.
    pub isinma_adimi: u64,
    /// Steps this call may take.
    pub toplam_adim: u64,
    /// Steps the schedule was laid out for: `baslangic_adim + toplam_adim` of a
    /// run that is being run in pieces, the same number as `toplam_adim` of one
    /// that is not.
    pub planlanan_adim: u64,
    /// Step the run starts at. Non-zero only when resuming.
    pub baslangic_adim: u64,
    /// Best validation loss carried into this call, if any.
    pub baslangic_en_iyi: Option<f64>,
    /// Last completed epoch carried into this call.
    pub baslangic_epoch: u32,
    /// How far into the next epoch the previous call got. A call that stopped
    /// in the middle of an epoch left windows unread; starting the epoch over
    /// would train on the first ones twice and reshuffle a run that is supposed
    /// to be the continuation of another.
    pub baslangic_konum: usize,
    /// Windows per step: gradients are averaged over their tokens.
    pub yigin: usize,
    /// Gradient-norm clip. Zero disables clipping, and says so in the report.
    pub kirpma: f64,
    /// Validate every this many steps.
    pub dogrulama_her: u64,
    /// Stop after this many epochs even if the loss is still improving.
    pub epoch_tavani: u32,
}

/// One training step, recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct AdimKaydi {
    /// Step number, counting from the start of the run (not of this call).
    pub adim: u64,
    /// Epoch number, counting from the start of the run.
    pub epoch: u32,
    /// Token-weighted mean loss over the windows in this step.
    pub kayip: f64,
    /// Learning rate used for this step.
    pub ogrenme_orani: f64,
    /// Gradient norm before clipping.
    pub gradyan_normu: f64,
    /// Whether the clip actually bit.
    pub kirpildi: bool,
    /// Tokens this step read.
    pub jeton: u64,
}

/// One validation pass, recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct DogrulamaKaydi {
    /// Step the measurement follows.
    pub adim: u64,
    /// Epoch the measurement follows.
    pub epoch: u32,
    /// Token-weighted mean loss over the validation windows.
    pub kayip: f64,
    /// Windows measured.
    pub pencere: usize,
    /// Tokens measured.
    pub jeton: u64,
}

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurmaNedeni {
    /// The step budget for this call ran out.
    AdimButcesi,
    /// An epoch failed to beat the best validation loss.
    EpochKurali,
    /// The epoch ceiling was reached.
    EpochTavani,
}

impl DurmaNedeni {
    /// The label the report carries.
    #[must_use]
    pub fn etiket(self) -> &'static str {
        match self {
            Self::AdimButcesi => "adim-butcesi",
            Self::EpochKurali => "epoch-kurali",
            Self::EpochTavani => "epoch-tavani",
        }
    }
}

/// Why a run was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KosuHatasi {
    /// A configuration that cannot mean anything: a zero step budget, a window
    /// outside the spec, a schedule horizon shorter than the run.
    GecersizAyari,
    /// Parameters whose blocks are not the spec's shape.
    SekilUyusmuyor,
    /// The optimizer's moment vectors do not cover the parameters.
    OptimizerUyusmuyor,
    /// No training windows.
    BosEgitim,
    /// No validation windows: a run that cannot be measured is not run.
    BosDogrulama,
}

/// What one call produced, enough to checkpoint, resume and compare.
#[derive(Debug, Clone, PartialEq)]
pub struct KosuRaporu {
    /// The configuration this run was given.
    pub ayar: KosuAyari,
    /// Last step taken.
    pub adim: u64,
    /// Last epoch reached.
    pub epoch: u32,
    /// Every training step's record.
    pub egitim_egrisi: Vec<AdimKaydi>,
    /// Every validation pass's record.
    pub dogrulama_egrisi: Vec<DogrulamaKaydi>,
    /// Best validation loss seen, carried-in value included.
    pub en_iyi_dogrulama: Option<f64>,
    /// Step the best validation loss was seen at.
    pub en_iyi_dogrulama_adimi: Option<u64>,
    /// First training loss in this call.
    pub baslangic_kaybi: f64,
    /// Last training loss in this call.
    pub son_kaybi: f64,
    /// Mean training loss per epoch.
    pub epoch_kaybi: Vec<f64>,
    /// Training tokens read.
    pub jeton: u64,
    /// Steps where clipping bit.
    pub kirpilan_adim: u64,
    /// Training windows available.
    pub egitim_pencere: usize,
    /// Validation windows available.
    pub dogrulama_pencere: usize,
    /// Why the loop stopped.
    pub durma_nedeni: DurmaNedeni,
    /// Wall clock, milliseconds. Machine-dependent; recorded, never ratcheted.
    pub sure_ms: u128,
    /// Where a continuation of this run should pick the epoch up. Zero unless
    /// the call stopped in the middle of an epoch.
    pub devam_konum: usize,
}

impl KosuRaporu {
    /// Steps this call actually took.
    #[must_use]
    pub fn harcanan_adim(&self) -> u64 {
        self.adim.saturating_sub(self.ayar.baslangic_adim)
    }

    /// The fall from the first loss to the last, as a share of the first.
    #[must_use]
    pub fn dusus_orani(&self) -> Option<f64> {
        if self.baslangic_kaybi > 0.0 && self.son_kaybi.is_finite() {
            return Some((self.baslangic_kaybi - self.son_kaybi) / self.baslangic_kaybi);
        }
        None
    }
}

/// The learning rate at a step: linear warm-up, then a cosine to a tenth.
///
/// The cosine is laid out over the *planned* horizon, so a run split into two
/// calls sees one curve, not two.
fn ogrenme_orani(ayar: &KosuAyari, adim: u64) -> f64 {
    let tepe = ayar.ogrenme_orani;
    if ayar.isinma_adimi > 0 && adim <= ayar.isinma_adimi {
        return tepe * (adim as f64) / (ayar.isinma_adimi as f64);
    }
    let son = ayar.planlanan_adim.max(ayar.isinma_adimi + 1);
    let ilerleme =
        ((adim.saturating_sub(ayar.isinma_adimi)) as f64) / ((son - ayar.isinma_adimi) as f64);
    let ilerleme = ilerleme.clamp(0.0, 1.0);
    let taban = 0.1 * tepe;
    taban + 0.5 * (tepe - taban) * (1.0 + (std::f64::consts::PI * ilerleme).cos())
}

/// A deterministic permutation of `0..n`.
///
/// Fisher-Yates over a splitmix64 stream, seeded by the epoch: the same seed and
/// the same epoch give the same order on every machine, which is the only way
/// two runs can be compared at all.
fn karisim(n: usize, tohum: u64, epoch: u32) -> Vec<usize> {
    let mut durum = tohum
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(u64::from(epoch).wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(0x94d0_49bb_1331_11eb);
    let mut sonraki = move || {
        durum = durum.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = durum;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut sira: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (sonraki() % (i as u64 + 1)) as usize;
        sira.swap(i, j);
    }
    sira
}

/// Input, target and provenance for one window.
///
/// Every position predicts the next token in the stream. Attention does not
/// cross a record boundary - the mask lives in the trainer and is measured
/// there - while the label at a boundary is the next token of the stream, which
/// is the rule the packing test states.
fn pencere_hedefleri(pencere: &crate::PaketPencere) -> (Vec<usize>, Vec<usize>, Vec<u32>) {
    let n = pencere.kimlikler.len();
    let girdi: Vec<usize> = pencere.kimlikler[..n - 1]
        .iter()
        .map(|j| *j as usize)
        .collect();
    let hedef: Vec<usize> = pencere.kimlikler[1..].iter().map(|j| *j as usize).collect();
    let kaynak: Vec<u32> = pencere.kaynak[..n - 1].to_vec();
    (girdi, hedef, kaynak)
}

/// Scale gradients down to a norm ceiling, returning the norm before clipping.
fn kirp(gradyan: &mut Parametreler, esik: f64) -> (f64, bool) {
    let mut toplam = 0.0;
    for (_, blok) in gradyan.bloklar_adli() {
        for deger in blok {
            toplam += deger * deger;
        }
    }
    let norm = toplam.sqrt();
    if esik <= 0.0 || !norm.is_finite() || norm <= esik {
        return (norm, false);
    }
    gradyan.olcekle(esik / norm);
    (norm, true)
}

/// Run one training pass over packed windows.
///
/// `parametreler` and `optimizer` are mutated in place; a checkpoint is written
/// by the caller from the state this function leaves behind, which is why the
/// run itself never touches the filesystem.
///
/// The loop shuffles the training windows once per epoch, averages the
/// gradients of `yigin` windows per step (weighted by tokens, so a short window
/// does not outweigh a long one), clips, and steps the optimizer once per step.
/// Validation runs every `dogrulama_her` steps and at the end of every epoch;
/// the epoch rule reads those numbers, not the training loss, because a training
/// loss that falls while validation rises is the one curve that lies.
///
/// # Errors
/// [`KosuHatasi`] when the configuration cannot mean anything, when the
/// parameters are not the spec's shape, or when the optimizer's moment vectors
/// are not the parameter count long.
pub fn egitim_kosu(
    ayar: &KosuAyari,
    egitim: &[crate::PaketPencere],
    dogrulama: &[crate::PaketPencere],
    parametreler: &mut Parametreler,
    optimizer: &mut Adamw,
    mut geri_bildirim: impl FnMut(&AdimKaydi, Option<&DogrulamaKaydi>),
) -> Result<KosuRaporu, KosuHatasi> {
    ayar.spec.dogrula().map_err(|_| KosuHatasi::GecersizAyari)?;
    if ayar.toplam_adim == 0
        || ayar.yigin == 0
        || ayar.dogrulama_her == 0
        || ayar.pencere_uzunlugu < 2
        || ayar.pencere_uzunlugu > ayar.spec.max_seq_len
        || ayar.epoch_tavani == 0
        || ayar.planlanan_adim < ayar.baslangic_adim + ayar.toplam_adim
    {
        return Err(KosuHatasi::GecersizAyari);
    }
    if !ayar.ogrenme_orani.is_finite() || ayar.ogrenme_orani <= 0.0 {
        return Err(KosuHatasi::GecersizAyari);
    }
    if !(ayar.agirlik_sonumu.is_finite() && (0.0..1.0).contains(&ayar.agirlik_sonumu)) {
        return Err(KosuHatasi::GecersizAyari);
    }
    if ayar.kirpma < 0.0 || ayar.kirpma.is_nan() {
        return Err(KosuHatasi::GecersizAyari);
    }
    if egitim.is_empty() {
        return Err(KosuHatasi::BosEgitim);
    }
    if dogrulama.is_empty() {
        return Err(KosuHatasi::BosDogrulama);
    }
    if !parametreler.sekil_dogru(ayar.spec) {
        return Err(KosuHatasi::SekilUyusmuyor);
    }
    let oge = parametreler.toplam_ogeler();
    let (_opt_adim, opt_m, opt_v) = optimizer.durum();
    if opt_m.len() != oge || opt_v.len() != oge {
        return Err(KosuHatasi::OptimizerUyusmuyor);
    }

    let baslangic = std::time::Instant::now();
    let sonum_maskesi = parametreler.sonum_maskesi();
    let mut egitim_egrisi: Vec<AdimKaydi> = Vec::new();
    let mut dogrulama_egrisi: Vec<DogrulamaKaydi> = Vec::new();
    let mut epoch_kaybi: Vec<f64> = Vec::new();
    let mut en_iyi: Option<f64> = ayar.baslangic_en_iyi;
    let mut en_iyi_adim: Option<u64> = None;
    let mut durma = DurmaNedeni::AdimButcesi;
    let mut adim = ayar.baslangic_adim;
    let mut tamamlanan_epoch: u32 = ayar.baslangic_epoch;
    let mut okunan_jeton: u64 = 0;
    let mut kirpilan: u64 = 0;
    let mut ilk_kayip: Option<f64> = None;
    let bu_cagrinin_tavani = ayar.baslangic_adim + ayar.toplam_adim;
    let mut baslangic_konum = ayar.baslangic_konum;

    while adim < bu_cagrinin_tavani {
        if tamamlanan_epoch + 1 > ayar.epoch_tavani {
            durma = DurmaNedeni::EpochTavani;
            break;
        }
        // Mutlak epoch: devam eden bir turun pencere sirasi, kesintisiz turun
        // ayni epoch'unda ayni olmali.
        let mutlak = tamamlanan_epoch + 1;
        let sira = karisim(egitim.len(), ayar.tohum, mutlak);
        let mut konum = baslangic_konum;
        let mut epoch_ici_kayip = 0.0f64;
        let mut epoch_ici_adim: u64 = 0;
        while konum < sira.len() && adim < bu_cagrinin_tavani {
            let oran = ogrenme_orani(ayar, adim + 1);
            optimizer.ogrenme_orani = oran;
            let mut toplam_gradyan = parametreler.sifir_gradyan();
            let mut toplam_kayip = 0.0f64;
            let mut toplam_jeton = 0usize;
            for _ in 0..ayar.yigin {
                if konum >= sira.len() {
                    break;
                }
                let pencere = &egitim[sira[konum]];
                konum += 1;
                let (girdi, hedef, kaynak) = pencere_hedefleri(pencere);
                let (kayip, grad) =
                    ileri_ve_geri_paket(ayar.spec, parametreler, &girdi, &hedef, &kaynak);
                toplam_kayip += kayip * girdi.len() as f64;
                toplam_jeton += girdi.len();
                toplam_gradyan.topla_ile(&grad);
            }
            if toplam_jeton == 0 {
                continue;
            }
            toplam_gradyan.olcekle(1.0 / toplam_jeton as f64);
            let (norm, kirpildi) = kirp(&mut toplam_gradyan, ayar.kirpma);
            if kirpildi {
                kirpilan += 1;
            }
            let kayip = toplam_kayip / toplam_jeton as f64;
            optimizer
                .adim_maskele(parametreler, &toplam_gradyan, &sonum_maskesi)
                .map_err(|_| KosuHatasi::OptimizerUyusmuyor)?;
            adim += 1;
            okunan_jeton += toplam_jeton as u64;
            epoch_ici_kayip += kayip;
            epoch_ici_adim += 1;
            if ilk_kayip.is_none() {
                ilk_kayip = Some(kayip);
            }

            let mut olcum: Option<DogrulamaKaydi> = None;
            if adim % ayar.dogrulama_her == 0 {
                let d = dogrula(ayar, dogrulama, parametreler, adim, mutlak);
                en_iyi = Some(en_iyi.map_or(d.kayip, |e| e.min(d.kayip)));
                if en_iyi_adim.is_none() || Some(d.kayip) == en_iyi {
                    en_iyi_adim = Some(adim);
                }
                dogrulama_egrisi.push(d.clone());
                olcum = Some(d);
            }
            let kayit = AdimKaydi {
                adim,
                epoch: mutlak,
                kayip,
                ogrenme_orani: oran,
                gradyan_normu: norm,
                kirpildi,
                jeton: toplam_jeton as u64,
            };
            geri_bildirim(&kayit, olcum.as_ref());
            egitim_egrisi.push(kayit);
        }
        if konum < sira.len() {
            // Epoch yarim kaldi: butce bitti. Devam eden tur buradan devam
            // edecek, o yuzden konum cagirana geri verilir.
            baslangic_konum = konum;
            break;
        }
        baslangic_konum = 0;
        tamamlanan_epoch += 1;
        if epoch_ici_adim > 0 {
            epoch_kaybi.push(epoch_ici_kayip / epoch_ici_adim as f64);
        }
        // Epoch kurali: tamamlanan bir epoch, devralinan en iyi dogrulamayi
        // gecemediyse tur durur. Karsilastirma dogrulama kaybi uzerinden
        // yapilir; egitim kaybi dusmeye devam ederken dogrulama kaybi
        // yukseliyorsa o egri yalan soyluyor demektir.
        let onceki_en_iyi = en_iyi;
        let d = dogrula(ayar, dogrulama, parametreler, adim, tamamlanan_epoch);
        let iyilesti = onceki_en_iyi.is_none_or(|e| d.kayip < e);
        if iyilesti {
            en_iyi = Some(d.kayip);
            en_iyi_adim = Some(adim);
        }
        dogrulama_egrisi.push(d);
        if adim >= bu_cagrinin_tavani {
            break;
        }
        if !iyilesti {
            durma = DurmaNedeni::EpochKurali;
            break;
        }
    }

    let son_kayip = egitim_egrisi.last().map_or(f64::NAN, |k| k.kayip);
    Ok(KosuRaporu {
        ayar: ayar.clone(),
        adim,
        epoch: tamamlanan_epoch,
        egitim_egrisi,
        dogrulama_egrisi,
        en_iyi_dogrulama: en_iyi,
        en_iyi_dogrulama_adimi: en_iyi_adim,
        baslangic_kaybi: ilk_kayip.unwrap_or(f64::NAN),
        son_kaybi: son_kayip,
        epoch_kaybi,
        jeton: okunan_jeton,
        kirpilan_adim: kirpilan,
        egitim_pencere: egitim.len(),
        dogrulama_pencere: dogrulama.len(),
        durma_nedeni: durma,
        sure_ms: baslangic.elapsed().as_millis(),
        devam_konum: baslangic_konum,
    })
}

/// One validation pass: no gradients, no updates, token-weighted mean loss.
fn dogrula(
    ayar: &KosuAyari,
    dogrulama: &[crate::PaketPencere],
    parametreler: &Parametreler,
    adim: u64,
    mutlak_epoch: u32,
) -> DogrulamaKaydi {
    let mut toplam_kayip = 0.0f64;
    let mut toplam_jeton = 0usize;
    for pencere in dogrulama {
        let (girdi, hedef, kaynak) = pencere_hedefleri(pencere);
        let (kayip, _) = ileri_ve_geri_paket(ayar.spec, parametreler, &girdi, &hedef, &kaynak);
        toplam_kayip += kayip * girdi.len() as f64;
        toplam_jeton += girdi.len();
    }
    DogrulamaKaydi {
        adim,
        epoch: mutlak_epoch,
        kayip: if toplam_jeton == 0 {
            f64::NAN
        } else {
            toplam_kayip / toplam_jeton as f64
        },
        pencere: dogrulama.len(),
        jeton: toplam_jeton as u64,
    }
}

/// Split records and cut windows in one step.
///
/// The CLI holds the corpus as records with token ids and the run wants windows;
/// the two steps are one sentence in every caller, and putting them together
/// here means the split rule is applied once.
///
/// # Errors
/// A string naming which step refused, with the reason.
pub fn bolumden_pencereler(
    kayitlar: Vec<Kayit>,
    dogrulama_payi: f64,
    uzunluk: usize,
) -> Result<(Bolum, Vec<crate::PaketPencere>, Vec<crate::PaketPencere>), String> {
    let bolum = crate::veri::bolumle(kayitlar, dogrulama_payi)
        .map_err(|h| format!("bolumleme reddedildi: {h:?}"))?;
    let (egitim, dogrulama) = crate::veri::pencereler(&bolum, uzunluk)?;
    Ok((bolum, egitim, dogrulama))
}

/// Pack a list of token sequences without a split; the measurement path uses it.
///
/// # Errors
/// [`PaketHatasi`] on a zero window or no records.
pub fn pencereleri_paketle(
    diziler: &[Vec<u32>],
    uzunluk: usize,
) -> Result<(Vec<crate::PaketPencere>, crate::PaketRaporu), crate::PaketHatasi> {
    paketle(diziler, uzunluk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BLOK_ADLARI, INIT_STD_EMBEDDING};

    fn kucuk_spec() -> Spec {
        Spec {
            vocab: 32,
            d_model: 16,
            n_layers: 2,
            n_heads: 2,
            d_ff: 32,
            max_seq_len: 8,
        }
    }

    fn pencereler(kayit_sayisi: usize, jeton: usize, uzunluk: usize) -> Vec<crate::PaketPencere> {
        let diziler: Vec<Vec<u32>> = (0..kayit_sayisi)
            .map(|k| {
                (0..jeton)
                    .map(|i| ((i + k * 3) % 29) as u32)
                    .collect::<Vec<u32>>()
            })
            .collect();
        let (p, _) = paketle(&diziler, uzunluk).expect("pack");
        p
    }

    fn ayar(spec: Spec, adim: u64) -> KosuAyari {
        KosuAyari {
            spec,
            pencere_uzunlugu: 8,
            tohum: 5,
            ogrenme_orani: 0.02,
            agirlik_sonumu: 0.1,
            isinma_adimi: 2,
            toplam_adim: adim,
            planlanan_adim: adim,
            baslangic_adim: 0,
            baslangic_en_iyi: None,
            baslangic_epoch: 0,
            baslangic_konum: 0,
            yigin: 1,
            kirpma: 1.0,
            dogrulama_her: 2,
            epoch_tavani: 4,
        }
    }

    fn hazir() -> (
        Spec,
        Parametreler,
        Adamw,
        Vec<crate::PaketPencere>,
        Vec<crate::PaketPencere>,
    ) {
        let spec = kucuk_spec();
        let p = Parametreler::mup_init(spec, 3, INIT_STD_EMBEDDING);
        let opt = Adamw::yeni(p.toplam_ogeler(), 0.02, 0.1).expect("optimizer");
        let egitim = pencereler(2, 24, 8);
        let dogrulama = pencereler(1, 24, 8);
        (spec, p, opt, egitim, dogrulama)
    }

    #[test]
    fn a_run_measured_its_own_descent() {
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        let once =
            crate::ileri_ve_geri_paket(spec, &p, &[0usize, 1, 2], &[1usize, 2, 3], &[0u32, 0, 0]).0;
        let rapor = egitim_kosu(
            &ayar(spec, 6),
            &egitim,
            &dogrulama,
            &mut p,
            &mut opt,
            |_, _| {},
        )
        .expect("run");
        assert_eq!(rapor.harcanan_adim(), 6);
        assert_eq!(rapor.egitim_egrisi.len(), 6);
        assert!(
            rapor.son_kaybi < once,
            "kayip dusmedi: {:.6} -> {:.6}",
            once,
            rapor.son_kaybi
        );
        assert!(rapor.dusus_orani().is_some_and(|d| d > 0.0));
        assert!(!rapor.dogrulama_egrisi.is_empty());
    }

    #[test]
    fn a_run_without_validation_is_refused() {
        let (spec, mut p, mut opt, egitim, _) = hazir();
        let bos: Vec<crate::PaketPencere> = Vec::new();
        assert_eq!(
            egitim_kosu(&ayar(spec, 2), &egitim, &bos, &mut p, &mut opt, |_, _| {}),
            Err(KosuHatasi::BosDogrulama)
        );
    }

    #[test]
    fn a_schedule_horizon_shorter_than_the_run_is_refused() {
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        let mut a = ayar(spec, 8);
        a.planlanan_adim = 4;
        assert_eq!(
            egitim_kosu(&a, &egitim, &dogrulama, &mut p, &mut opt, |_, _| {}),
            Err(KosuHatasi::GecersizAyari)
        );
    }

    #[test]
    fn a_zero_window_or_zero_budget_is_refused() {
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        for bozuk in [
            KosuAyari {
                toplam_adim: 0,
                ..ayar(spec, 4)
            },
            KosuAyari {
                yigin: 0,
                ..ayar(spec, 4)
            },
            KosuAyari {
                pencere_uzunlugu: 0,
                ..ayar(spec, 4)
            },
            KosuAyari {
                pencere_uzunlugu: 9,
                ..ayar(spec, 4)
            },
            KosuAyari {
                epoch_tavani: 0,
                ..ayar(spec, 4)
            },
            KosuAyari {
                ogrenme_orani: 0.0,
                ..ayar(spec, 4)
            },
            KosuAyari {
                agirlik_sonumu: 1.0,
                ..ayar(spec, 4)
            },
            KosuAyari {
                kirpma: -1.0,
                ..ayar(spec, 4)
            },
        ] {
            assert_eq!(
                egitim_kosu(&bozuk, &egitim, &dogrulama, &mut p, &mut opt, |_, _| {}),
                Err(KosuHatasi::GecersizAyari),
                "bozuk ayar kabul edildi: {bozuk:?}"
            );
        }
    }

    #[test]
    fn a_resumed_run_reproduces_an_uninterrupted_one() {
        // Karar verici olcum: 6+6 adim, 12 adimin aynisi olmali - kayip
        // egrisi, adim adim, bit bit ayni. Devam eden turun epoch'u, optimizer
        // adimi ve LR ufku tasinmazsa bu esitlik bozulur.
        let spec = kucuk_spec();
        // Epoch, 12 adimdan uzun: bu test epoch ortasinda kesilen bir turun
        // ayni yerden devam ettigini olcuyor.
        let egitim = pencereler(8, 40, 8);
        let dogrulama = pencereler(2, 24, 8);
        let tum = KosuAyari {
            planlanan_adim: 12,
            dogrulama_her: 3,
            isinma_adimi: 2,
            ..ayar(spec, 12)
        };
        let mut p = Parametreler::mup_init(spec, 11, INIT_STD_EMBEDDING);
        let mut opt = Adamw::yeni(p.toplam_ogeler(), 0.02, 0.1).expect("optimizer");
        let tam = egitim_kosu(&tum, &egitim, &dogrulama, &mut p, &mut opt, |_, _| {}).expect("run");

        let mut p2 = Parametreler::mup_init(spec, 11, INIT_STD_EMBEDDING);
        let mut opt2 = Adamw::yeni(p2.toplam_ogeler(), 0.02, 0.1).expect("optimizer");
        let yari = KosuAyari {
            toplam_adim: 6,
            ..tum.clone()
        };
        let birinci =
            egitim_kosu(&yari, &egitim, &dogrulama, &mut p2, &mut opt2, |_, _| {}).expect("run 1");
        let devam = KosuAyari {
            toplam_adim: 6,
            planlanan_adim: 12,
            baslangic_adim: birinci.adim,
            baslangic_epoch: birinci.epoch,
            baslangic_en_iyi: birinci.en_iyi_dogrulama,
            baslangic_konum: birinci.devam_konum,
            ..tum.clone()
        };
        let ikinci =
            egitim_kosu(&devam, &egitim, &dogrulama, &mut p2, &mut opt2, |_, _| {}).expect("run 2");

        let mut tum_kayip: Vec<f64> = tam.egitim_egrisi.iter().map(|k| k.kayip).collect();
        let mut parca: Vec<f64> = birinci.egitim_egrisi.iter().map(|k| k.kayip).collect();
        parca.extend(ikinci.egitim_egrisi.iter().map(|k| k.kayip));
        assert_eq!(tum_kayip.len(), parca.len(), "adim sayisi tutmuyor");
        for (a, b) in tum_kayip.drain(..).zip(parca) {
            assert!(
                (a - b).abs() < 1e-12,
                "kayip egrisi kaydi: {a:.12} vs {b:.12}"
            );
        }
        assert!(
            (p.embedding[0] - p2.embedding[0]).abs() < 1e-12,
            "devam eden tur baska agirliklara vardi"
        );
    }

    #[test]
    fn the_epoch_rule_stops_a_run_that_stops_improving() {
        // Dogrulama kaybi iyilesmiyorsa tur, adim butcesi dolmadan durur ve
        // nedenini soyler.
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        let mut a = ayar(spec, 20);
        a.dogrulama_her = 1;
        a.baslangic_en_iyi = Some(0.0);
        let rapor = egitim_kosu(&a, &egitim, &dogrulama, &mut p, &mut opt, |_, _| {}).expect("run");
        assert_eq!(rapor.durma_nedeni, DurmaNedeni::EpochKurali);
        assert_eq!(rapor.durma_nedeni.etiket(), "epoch-kurali");
        assert!(rapor.epoch >= 1);
        assert!(rapor.harcanan_adim() < 20, "erken durmadi");
    }

    #[test]
    fn the_epoch_ceiling_stops_a_run_that_keeps_improving() {
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        let mut a = ayar(spec, 40);
        a.epoch_tavani = 1;
        a.dogrulama_her = 20;
        let rapor = egitim_kosu(&a, &egitim, &dogrulama, &mut p, &mut opt, |_, _| {}).expect("run");
        assert_eq!(rapor.durma_nedeni, DurmaNedeni::EpochTavani);
        assert!(rapor.harcanan_adim() <= 40);
    }

    #[test]
    fn the_report_names_every_step_it_took() {
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        let rapor = egitim_kosu(
            &ayar(spec, 5),
            &egitim,
            &dogrulama,
            &mut p,
            &mut opt,
            |_, _| {},
        )
        .expect("run");
        let jeton: u64 = rapor.egitim_egrisi.iter().map(|k| k.jeton).sum();
        assert_eq!(jeton, rapor.jeton);
        let adimlar: Vec<u64> = rapor.egitim_egrisi.iter().map(|k| k.adim).collect();
        assert_eq!(adimlar, vec![1, 2, 3, 4, 5]);
        assert!(rapor.egitim_egrisi.iter().all(|k| k.ogrenme_orani > 0.0));
        assert!(rapor.epoch_kaybi.iter().all(|k| k.is_finite()));
    }

    #[test]
    fn the_feedback_receives_the_step_and_its_measurement() {
        let (spec, mut p, mut opt, egitim, dogrulama) = hazir();
        let mut gorulen: Vec<(u64, bool)> = Vec::new();
        egitim_kosu(
            &ayar(spec, 4),
            &egitim,
            &dogrulama,
            &mut p,
            &mut opt,
            |k, d| {
                gorulen.push((k.adim, d.is_some()));
            },
        )
        .expect("run");
        assert_eq!(gorulen.len(), 4);
        assert!(
            gorulen.iter().any(|(_, olcum)| *olcum),
            "dogrulama hic bildirilmedi"
        );
    }

    #[test]
    fn the_split_and_the_windows_come_from_one_call() {
        let kayitlar: Vec<Kayit> = (0..12)
            .map(|i| Kayit {
                kimlik: format!("k-{i}"),
                jetonlar: (0..40).map(|j| ((i + j) % 29) as u32).collect(),
            })
            .collect();
        let (bolum, egitim, dogrulama) = bolumden_pencereler(kayitlar, 0.25, 8).expect("split");
        assert_eq!(bolum.egitim.len() + bolum.dogrulama.len(), 12);
        assert!(!egitim.is_empty() && !dogrulama.is_empty());
        assert!(bolumden_pencereler(Vec::new(), 0.25, 8).is_err());
        assert!(bolumden_pencereler(
            vec![Kayit {
                kimlik: "a".into(),
                jetonlar: (0..40).map(|i| i as u32).collect()
            }],
            0.25,
            8
        )
        .is_err());
    }

    #[test]
    fn the_parameter_blocks_are_the_ones_the_format_names() {
        let spec = kucuk_spec();
        let p = Parametreler::sifir(spec);
        let adlar: Vec<&str> = p.bloklar_adli().iter().map(|(a, _)| *a).collect();
        assert_eq!(adlar.len(), BLOK_ADLARI.len());
        assert_eq!(adlar, Parametreler::blok_adlari());
        assert_eq!(p.bloklar().len(), 19);
        let maske = p.sonum_maskesi();
        assert_eq!(maske.len(), p.toplam_ogeler());
        assert!(p.sekil_dogru(spec));
        let mut kopya = p.clone();
        let mut blok = vec![0.0; kopya.b1.len()];
        blok[0] = 1.0;
        assert!(kopya.blok_ata("b1", blok));
        assert_eq!(kopya.b1[0], 1.0);
        assert!(!kopya.blok_ata("boyle-bir-blok-yok", vec![1.0]));
        assert!(!kopya.blok_ata("b1", vec![1.0, 2.0]));
    }
}
