//! # lubot-cikarim - reading a trained checkpoint: score and rank, never generate
//!
//! This crate is the inference surface. It loads a checkpoint written by
//! `lubot-egitim`, runs the network forward, and answers two questions:
//! *how well does this checkpoint predict this text, given that context*, and
//! *which of these candidates does it prefer*. This crate still has no decoder
//! loop and produces no text: scoring and ranking are its contract, and the CLI
//! has a gate that fails if a generation surface appears.
//!
//! [`ornekleyici`] carries the *distribution* half of sampling as pure
//! mathematics - temperature, top-k, nucleus, and a deterministic stream - so
//! that the arithmetic is testable on its own. It holds no loop, no network and
//! no entry point: turning a distribution into a written answer is a separate
//! surface with its own decision behind it.
//!
//! # The one hard rule: score a token from before it
//!
//! The hidden state after position `j` predicts the token at `j+1`, never the
//! token at `j`. Scoring a token from a state that already contains it is the
//! leakage that makes any model look better than it is, and it is invisible:
//! the number is still a plausible-looking log-probability. [`Cikarim::puanla`]
//! therefore advances the cache first and reads the probability second, and a
//! request that would have nothing to score - an empty context with a single
//! token - is refused instead of answered with a vocabulary prior.
//!
//! # Why the two passes are compared
//!
//! The fast path keeps a key/value cache and reads one position at a time; the
//! slow path re-reads every prefix from scratch for every position. They must
//! agree to within [`CACHE_TOLERANCE`], and the training kernel's own loss is
//! the third opinion. An optimisation that changes the answer is a bug with a
//! speed claim attached, so the check is a measurement with a tolerance and not
//! a comment saying "equivalent".

pub mod cezalar;
pub mod motor;
pub mod ornekleyici;
pub mod uretim;

use std::path::Path;

use lubot_egitim::kontrol::{Kontrol, KontrolHatasi, OptimizerDurumu};
use lubot_egitim::{ileri_ve_geri_paket, Parametreler, Spec, LN_EPS};

/// Largest absolute difference the cached and recomputed paths may show.
///
/// Measured on the self corpus, the two agree to around 1e-15; the tolerance is
/// three orders of magnitude above that, because a threshold set at the noise
/// floor fails on a different machine's rounding rather than on a real defect.
pub const CACHE_TOLERANCE: f64 = 1e-9;

/// Why scoring or loading was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum CikarimHatasi {
    /// The checkpoint was refused by its own loader.
    Kontrol(String),
    /// A checkpoint with no optimizer state cannot be continued, but it can be
    /// scored; this is the refusal for a file that is not a checkpoint at all.
    BosGirdi,
    /// A token id outside the vocabulary.
    KimlikAraligi { jeton: u32, sozluk: usize },
    /// Context plus text longer than the window the spec was built for.
    PencereAsimi { istenen: usize, tavan: usize },
    /// A ranking request with no candidates.
    BosAday,
}

impl std::fmt::Display for CikarimHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kontrol(s) => write!(f, "kontrol noktasi reddedildi: {s}"),
            Self::BosGirdi => write!(f, "puanlanacak jeton yok"),
            Self::KimlikAraligi { jeton, sozluk } => {
                write!(f, "jeton {jeton} sozluk disinda (0..{sozluk})")
            }
            Self::PencereAsimi { istenen, tavan } => {
                write!(f, "pencere asimi: {istenen} jeton, tavan {tavan}")
            }
            Self::BosAday => write!(f, "siralanacak aday yok"),
        }
    }
}

/// What the cache check measured.
#[derive(Debug, Clone, PartialEq)]
pub struct OnbellekRaporu {
    /// Positions compared.
    pub konum: usize,
    /// Mean log-probability from the cached path.
    pub onbellekli: f64,
    /// Mean log-probability from the path that re-reads every prefix.
    pub tam_gecis: f64,
    /// Mean log-probability implied by the training kernel's loss.
    pub egitim_cekirdegi: f64,
    /// Largest per-position gap between the cached path and the recomputation.
    pub en_buyuk_fark_onbellek: f64,
    /// Largest gap between the cached path and the training kernel.
    pub en_buyuk_fark_egitim: f64,
}

/// One ranked candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Aday {
    /// Index of the candidate in the caller's list.
    pub sira: usize,
    /// Mean log-probability per scored token; higher is better.
    pub puan: f64,
    /// Tokens that were scored.
    pub jeton: usize,
}

/// A checkpoint loaded for inference.
#[derive(Debug, Clone)]
pub struct Cikarim {
    spec: Spec,
    parametreler: Parametreler,
    adim: u64,
    sozluk_aile: String,
    korpus_ozeti: String,
    en_iyi_dogrulama: Option<f64>,
    optimizer: Option<OptimizerDurumu>,
}

/// Per-layer keys and values for the positions read so far.
#[derive(Debug, Clone)]
struct Onbellek {
    k: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    uzunluk: usize,
}

impl Onbellek {
    fn yeni(spec: Spec) -> Self {
        Self {
            k: vec![Vec::new(); spec.n_layers],
            v: vec![Vec::new(); spec.n_layers],
            uzunluk: 0,
        }
    }

    /// En eski `n` konumu önbellekten düşürür: kayan pencerenin hızlı yolu.
    ///
    /// Bu bir **yaklaşımdır** ve adı öyle yazılıdır: düşen konumların ardından
    /// gelen konumların gizli durumları yeniden hesaplanmaz - onlar daha uzun
    /// bir pencere görülerek hesaplanmıştı. Kayan pencere bundan sonraki
    /// dikkat için kısalır, geçmiş için değil. Tam yeniden hesaplayan yol
    /// [`crate::uretim::KaydirmaModu::YenidenKur`]'dur; hangisinin koştuğu
    /// üretim raporunda yazılıdır.
    fn bastan_dus(&mut self, n: usize, d_model: usize) {
        let n = n.min(self.uzunluk);
        if n == 0 {
            return;
        }
        for katman in 0..self.k.len() {
            let kes = (n * d_model).min(self.k[katman].len());
            self.k[katman].drain(..kes);
            self.v[katman].drain(..kes);
        }
        self.uzunluk -= n;
    }
}

impl Cikarim {
    /// Load a checkpoint.
    ///
    /// # Errors
    /// [`CikarimHatasi::Kontrol`] with the loader's reason: a damaged file, a
    /// version this build does not know, a digest that does not match.
    pub fn yukle(yol: &Path) -> Result<Self, CikarimHatasi> {
        let kontrol = Kontrol::yukle(yol)
            .map_err(|e: KontrolHatasi| CikarimHatasi::Kontrol(e.to_string()))?;
        Ok(Self::kontrollden(kontrol))
    }

    /// The same, from a checkpoint already in memory.
    #[must_use]
    pub fn kontrollden(kontrol: Kontrol) -> Self {
        Self {
            spec: kontrol.spec,
            parametreler: kontrol.parametreler,
            adim: kontrol.adim,
            sozluk_aile: kontrol.sozluk_aile,
            korpus_ozeti: kontrol.korpus_ozeti,
            en_iyi_dogrulama: kontrol.en_iyi_dogrulama,
            optimizer: kontrol.optimizer,
        }
    }

    /// The architecture this checkpoint was trained with.
    #[must_use]
    pub fn spec(&self) -> Spec {
        self.spec
    }

    /// Vocabulary size the checkpoint expects.
    #[must_use]
    pub fn sozluk_boyutu(&self) -> usize {
        self.spec.vocab
    }

    /// Vocabulary family the run tokenised with.
    #[must_use]
    pub fn sozluk_aile(&self) -> &str {
        &self.sozluk_aile
    }

    /// Digest of the records the run read.
    #[must_use]
    pub fn korpus_ozeti(&self) -> &str {
        &self.korpus_ozeti
    }

    /// Steps the run took.
    #[must_use]
    pub fn adim(&self) -> u64 {
        self.adim
    }

    /// Best validation loss the run reached, if it measured one.
    #[must_use]
    pub fn en_iyi_dogrulama(&self) -> Option<f64> {
        self.en_iyi_dogrulama
    }

    /// Whether the checkpoint carries optimizer state, i.e. whether a run could
    /// continue from it.
    #[must_use]
    pub fn devam_edilebilir(&self) -> bool {
        self.optimizer.is_some()
    }

    /// The mean log-probability per token of `metin`, given `baglam`.
    ///
    /// This is the model's answer to "how expected is this text, after that
    /// context" - the quantity a re-ranker needs and the only quantity this
    /// crate produces. Higher (closer to zero) means better predicted.
    ///
    /// # What counts as a token to score
    ///
    /// Only tokens that *have* a context. The first token of the whole sequence
    /// is predicted from nothing, and folding that number into the mean would
    /// mix "how surprising is this text" with "how frequent is this token in
    /// the vocabulary" - two different questions, one average. So a call with
    /// an empty context and a single token is refused rather than answered with
    /// a vocabulary prior.
    ///
    /// # Errors
    /// [`CikarimHatasi::BosGirdi`], [`CikarimHatasi::KimlikAraligi`],
    /// [`CikarimHatasi::PencereAsimi`].
    pub fn puanla(&self, baglam: &[u32], metin: &[u32]) -> Result<f64, CikarimHatasi> {
        if baglam.is_empty() && metin.is_empty() {
            return Err(CikarimHatasi::BosGirdi);
        }
        let toplam_uzunluk = baglam.len() + metin.len();
        if toplam_uzunluk > self.spec.max_seq_len {
            return Err(CikarimHatasi::PencereAsimi {
                istenen: toplam_uzunluk,
                tavan: self.spec.max_seq_len,
            });
        }
        self.kimlikleri_denetle(baglam)?;
        self.kimlikleri_denetle(metin)?;
        let mut onbellek = Onbellek::yeni(self.spec);
        let mut son_gizli: Option<Vec<f64>> = None;
        for jeton in baglam {
            son_gizli = Some(self.ileri_konum(*jeton, &mut onbellek)?);
        }
        let mut toplam = 0.0f64;
        let mut sayilan = 0usize;
        for jeton in metin {
            if let Some(gizli) = &son_gizli {
                toplam += self.log_olasilik(gizli, *jeton);
                sayilan += 1;
            }
            son_gizli = Some(self.ileri_konum(*jeton, &mut onbellek)?);
        }
        if sayilan == 0 {
            return Err(CikarimHatasi::BosGirdi);
        }
        Ok(toplam / sayilan as f64)
    }

    /// Score candidates against one context and return them best first.
    ///
    /// Ties are broken by the candidate's index, so the order is a function of
    /// the inputs and not of the sort implementation's mood; two candidates
    /// that score equally are *equal*, and saying which is better would be
    /// inventing a difference.
    ///
    /// # Errors
    /// [`CikarimHatasi::BosAday`] with no candidates; otherwise whatever
    /// [`Cikarim::puanla`] refuses, named with the candidate's index.
    pub fn pasaj_sirala(
        &self,
        baglam: &[u32],
        adaylar: &[Vec<u32>],
    ) -> Result<Vec<Aday>, CikarimHatasi> {
        if adaylar.is_empty() {
            return Err(CikarimHatasi::BosAday);
        }
        self.kimlikleri_denetle(baglam)?;
        let mut puanlanan: Vec<Aday> = Vec::with_capacity(adaylar.len());
        for (sira, aday) in adaylar.iter().enumerate() {
            // Uzun aday pencereye sigmiyorsa kuyrugundan kirpilir: soru ve
            // baglam one dir, kesilecek yer adayin sonudur.
            let yer = self.spec.max_seq_len.saturating_sub(baglam.len()).max(1);
            let kesilmis: Vec<u32> = if aday.len() > yer {
                aday[aday.len() - yer..].to_vec()
            } else {
                aday.clone()
            };
            let puan = self.puanla(baglam, &kesilmis).map_err(|e| match e {
                CikarimHatasi::BosGirdi => CikarimHatasi::BosGirdi,
                digeri => digeri,
            })?;
            puanlanan.push(Aday {
                sira,
                puan,
                jeton: kesilmis
                    .len()
                    .saturating_sub(if baglam.is_empty() { 1 } else { 0 }),
            });
        }
        puanlanan.sort_by(|a, b| {
            b.puan
                .partial_cmp(&a.puan)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.sira.cmp(&b.sira))
        });
        Ok(puanlanan)
    }

    /// Measure the cached path, an independent recomputation, and the training
    /// kernel on the same ids.
    ///
    /// Three opinions, not two: the cached pass against a full pass catches a
    /// cache bug, and the training kernel catches the case where both this
    /// crate's paths share a mistake - which is exactly what a comparison
    /// against itself cannot see.
    ///
    /// # Errors
    /// [`CikarimHatasi::BosGirdi`] under two ids, plus the refusals
    /// [`Cikarim::puanla`] raises.
    pub fn onbellek_denetimi(&self, kimlikler: &[u32]) -> Result<OnbellekRaporu, CikarimHatasi> {
        if kimlikler.len() < 2 {
            return Err(CikarimHatasi::BosGirdi);
        }
        if kimlikler.len() > self.spec.max_seq_len {
            return Err(CikarimHatasi::PencereAsimi {
                istenen: kimlikler.len(),
                tavan: self.spec.max_seq_len,
            });
        }
        self.kimlikleri_denetle(kimlikler)?;
        let (onbellekli_dizi, tam_dizi) = self.iki_gecis(kimlikler)?;
        let onbellekli = ortalama(&onbellekli_dizi);
        let tam = ortalama(&tam_dizi);
        let mut en_buyuk = 0.0f64;
        for (a, b) in onbellekli_dizi.iter().zip(&tam_dizi) {
            en_buyuk = en_buyuk.max((a - b).abs());
        }
        // Egitim cekirdegi ayni diziyi kayip olarak okur: kayip =
        // -ortalama(log p). Ayni sayinin iki okunusu, iki ayri olcum degil.
        let girdi: Vec<usize> = kimlikler[..kimlikler.len() - 1]
            .iter()
            .map(|k| *k as usize)
            .collect();
        let hedef: Vec<usize> = kimlikler[1..].iter().map(|k| *k as usize).collect();
        let kaynak = vec![0u32; girdi.len()];
        let (kayip, _) =
            ileri_ve_geri_paket(self.spec, &self.parametreler, &girdi, &hedef, &kaynak);
        let egitim = -kayip;
        Ok(OnbellekRaporu {
            konum: tam_dizi.len(),
            onbellekli,
            tam_gecis: tam,
            egitim_cekirdegi: egitim,
            en_buyuk_fark_onbellek: en_buyuk,
            en_buyuk_fark_egitim: (onbellekli - egitim).abs(),
        })
    }

    /// The cached pass and a pass that re-reads every prefix, position by
    /// position, over the same ids.
    fn iki_gecis(&self, kimlikler: &[u32]) -> Result<(Vec<f64>, Vec<f64>), CikarimHatasi> {
        // Artimli yol: tek onbellek, her konum bir kez islenir.
        let mut onbellek = Onbellek::yeni(self.spec);
        let mut artimli: Vec<f64> = Vec::with_capacity(kimlikler.len() - 1);
        let mut son_gizli = self.ileri_konum(kimlikler[0], &mut onbellek)?;
        for jeton in &kimlikler[1..] {
            artimli.push(self.log_olasilik(&son_gizli, *jeton));
            son_gizli = self.ileri_konum(*jeton, &mut onbellek)?;
        }
        // Tam yol: her konum icin onek sifirdan yeniden islenir. Onbellegi
        // temizleyip ayni donguyu kosmak "onbellek kendisiyle ayni" demek
        // olurdu; burada bagimsiz bir hesap var.
        let mut tam: Vec<f64> = Vec::with_capacity(kimlikler.len() - 1);
        for konum in 1..kimlikler.len() {
            let mut yerel = Onbellek::yeni(self.spec);
            let mut gizli = self.ileri_konum(kimlikler[0], &mut yerel)?;
            for jeton in &kimlikler[1..konum] {
                gizli = self.ileri_konum(*jeton, &mut yerel)?;
            }
            tam.push(self.log_olasilik(&gizli, kimlikler[konum]));
        }
        Ok((artimli, tam))
    }

    /// One position through the network, returning the final hidden state.
    ///
    /// The hidden state - not the logits - is what comes back, because the
    /// logits are a dot product against the tied embedding: recomputing one row
    /// for one target costs `O(d)`, while materialising all `vocab` rows per
    /// position allocates a matrix to look at one number.
    ///
    /// The token is appended to the cache as a side effect: the cache *is* the
    /// context, and reading a position twice would attend over it twice.
    ///
    /// **The hidden state after `jeton` predicts the token that comes next.**
    /// It does not predict `jeton` itself: scoring a token from a state that
    /// already contains it is exactly the leakage this boundary prevents.
    fn ileri_konum(&self, jeton: u32, onbellek: &mut Onbellek) -> Result<Vec<f64>, CikarimHatasi> {
        let d = self.spec.d_model;
        let dk = self.spec.d_k();
        let mut x: Vec<f64> =
            self.parametreler.embedding[jeton as usize * d..(jeton as usize + 1) * d].to_vec();
        for katman in 0..self.spec.n_layers {
            let (ln1, _, _) = self.layer_norm(&x, katman, 1);
            let q = self.matmul(
                &ln1,
                &self.parametreler.wq,
                &self.parametreler.bq,
                katman,
                d,
            );
            let k = self.matmul(
                &ln1,
                &self.parametreler.wk,
                &self.parametreler.bk,
                katman,
                d,
            );
            let v = self.matmul(
                &ln1,
                &self.parametreler.wv,
                &self.parametreler.bv,
                katman,
                d,
            );
            onbellek.k[katman].extend_from_slice(&k);
            onbellek.v[katman].extend_from_slice(&v);
            let attn = self.dikkat_konum(&q, katman, onbellek, dk);
            let cikti = self.matmul(
                &attn,
                &self.parametreler.wo,
                &self.parametreler.bo,
                katman,
                d,
            );
            for (xi, c) in x.iter_mut().zip(&cikti) {
                *xi += c;
            }
            let (ln2, _, _) = self.layer_norm(&x, katman, 2);
            let w1 = &self.parametreler.w1
                [katman * self.spec.d_ff * d..(katman + 1) * self.spec.d_ff * d];
            let b1 = &self.parametreler.b1[katman * self.spec.d_ff..(katman + 1) * self.spec.d_ff];
            let mut gizli = vec![0.0f64; self.spec.d_ff];
            for (o, h) in gizli.iter_mut().enumerate() {
                let mut toplam = b1[o];
                let satir = &w1[o * d..o * d + d];
                for (m, wm) in satir.iter().enumerate() {
                    toplam += wm * ln2[m];
                }
                *h = gelu(toplam);
            }
            let w2 = &self.parametreler.w2
                [katman * d * self.spec.d_ff..(katman + 1) * d * self.spec.d_ff];
            let b2 = &self.parametreler.b2[katman * d..(katman + 1) * d];
            for (o, xo) in x.iter_mut().enumerate() {
                let mut toplam = b2[o];
                let satir = &w2[o * self.spec.d_ff..o * self.spec.d_ff + self.spec.d_ff];
                for (j, hj) in gizli.iter().enumerate() {
                    toplam += satir[j] * hj;
                }
                *xo += toplam;
            }
        }
        onbellek.uzunluk += 1;
        let (lnf, _, _) = self.layer_norm(&x, 0, 0);
        Ok(lnf)
    }

    /// Attention for the newest position over everything the cache holds.
    ///
    /// Causal by construction: the cache only ever contains positions up to and
    /// including the current one, so there is no future to mask. The summation
    /// order matches the trainer's, because a mean log-probability that agrees
    /// with the training loss to fifteen digits is what makes the third opinion
    /// in [`Cikarim::onbellek_denetimi`] worth having.
    fn dikkat_konum(&self, q: &[f64], katman: usize, onbellek: &Onbellek, dk: usize) -> Vec<f64> {
        let d = self.spec.d_model;
        let h = self.spec.n_heads;
        let t = onbellek.uzunluk + 1; // guncel konum dahil
        let k = &onbellek.k[katman];
        let v = &onbellek.v[katman];
        let olcek = 1.0 / (dk as f64).sqrt();
        let mut cikti = vec![0.0f64; d];
        for head in 0..h {
            let mut skor = vec![0.0f64; t];
            for (j, skor_j) in skor.iter_mut().enumerate() {
                let mut toplam = 0.0;
                for m in 0..dk {
                    toplam += q[head * dk + m] * k[j * d + head * dk + m];
                }
                *skor_j = toplam * olcek;
            }
            let yumusak = softmax(&skor);
            for m in 0..dk {
                let mut toplam = 0.0;
                for (j, w) in yumusak.iter().enumerate() {
                    toplam += w * v[j * d + head * dk + m];
                }
                cikti[head * dk + m] = toplam;
            }
        }
        cikti
    }

    /// The full logit vector for one hidden state: the tied readout with the
    /// spec's `1/d_model` scale.
    ///
    /// Scoring only ever needs two numbers (the target's logit and the maximum),
    /// which is why [`Cikarim::log_olasilik`] walks the vocabulary twice instead
    /// of materialising it. A sampler needs *all* of them, so this is the one
    /// place where the 8192-wide row is built - and it is built from the same
    /// embedding rows, in the same order, with the same scale, so a distribution
    /// taken from here and a log-probability taken from there cannot disagree.
    #[must_use]
    pub fn logitler(&self, gizli: &[f64]) -> Vec<f64> {
        let d = self.spec.d_model;
        let olcek = 1.0 / d as f64;
        let mut logitler = vec![0.0f64; self.spec.vocab];
        for (v, logit) in logitler.iter_mut().enumerate() {
            let satir = &self.parametreler.embedding[v * d..v * d + d];
            let mut toplam = 0.0;
            for (m, wm) in satir.iter().enumerate() {
                toplam += wm * gizli[m];
            }
            *logit = toplam * olcek;
        }
        logitler
    }

    /// `log p(hedef | hidden)`: the tied readout with the spec's `1/d_model`
    /// scale, then log-softmax picked at one index.
    fn log_olasilik(&self, gizli: &[f64], hedef: u32) -> f64 {
        let d = self.spec.d_model;
        let olcek = 1.0 / d as f64;
        // Iki gecis: once en buyuk logit ve hedefin logiti (pay icin), sonra
        // toplam. Ara dizi yok - sozluk 8192, d 64: tam logit vektoru her konum
        // icin 64 KiB olurdu ve tek bir sayi icin tumunu tutmak anlamsiz.
        let mut en_buyuk = f64::NEG_INFINITY;
        let mut hedef_logit = f64::NEG_INFINITY;
        for v in 0..self.spec.vocab {
            let satir = &self.parametreler.embedding[v * d..v * d + d];
            let mut toplam_logit = 0.0;
            for (m, wm) in satir.iter().enumerate() {
                toplam_logit += wm * gizli[m];
            }
            let logit = toplam_logit * olcek;
            if logit > en_buyuk {
                en_buyuk = logit;
            }
            if v == hedef as usize {
                hedef_logit = logit;
            }
        }
        if !en_buyuk.is_finite() || !hedef_logit.is_finite() {
            return f64::NEG_INFINITY;
        }
        let mut toplam = 0.0f64;
        for v in 0..self.spec.vocab {
            let satir = &self.parametreler.embedding[v * d..v * d + d];
            let mut toplam_logit = 0.0;
            for (m, wm) in satir.iter().enumerate() {
                toplam_logit += wm * gizli[m];
            }
            toplam += ((toplam_logit * olcek) - en_buyuk).exp();
        }
        // Egitim cekirdegi softmax olasiligini alip ln'ini aliyor; ayni yol,
        // ayni yuvarlama.
        let p = ((hedef_logit - en_buyuk).exp()) / toplam;
        p.ln()
    }

    fn matmul(&self, x: &[f64], w: &[f64], b: &[f64], katman: usize, d: usize) -> Vec<f64> {
        let ofset = katman * d * d;
        let mut y = vec![0.0f64; d];
        for (o, yo) in y.iter_mut().enumerate() {
            let mut toplam = b[katman * d + o];
            let satir = &w[ofset + o * d..ofset + o * d + d];
            for (m, wm) in satir.iter().enumerate() {
                toplam += wm * x[m];
            }
            *yo = toplam;
        }
        y
    }

    /// LayerNorm for one position. `katman` and `hangi` pick the block:
    /// `(l, 1)`, `(l, 2)`, or `(0, 0)` for the final norm.
    fn layer_norm(&self, x: &[f64], katman: usize, hangi: u8) -> (Vec<f64>, f64, f64) {
        let d = self.spec.d_model;
        let (olcek, sapma) = match hangi {
            1 => (
                &self.parametreler.ln1_olcek[katman * d..(katman + 1) * d],
                &self.parametreler.ln1_sapma[katman * d..(katman + 1) * d],
            ),
            2 => (
                &self.parametreler.ln2_olcek[katman * d..(katman + 1) * d],
                &self.parametreler.ln2_sapma[katman * d..(katman + 1) * d],
            ),
            _ => (
                &self.parametreler.lnf_olcek[..],
                &self.parametreler.lnf_sapma[..],
            ),
        };
        let mut toplam = 0.0;
        for deger in x {
            toplam += deger;
        }
        let ortalama = toplam / d as f64;
        let mut varyans = 0.0;
        for deger in x {
            let fark = deger - ortalama;
            varyans += fark * fark;
        }
        varyans /= d as f64;
        let rstd = 1.0 / (varyans + LN_EPS).sqrt();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(j, deger)| (deger - ortalama) * rstd * olcek[j] + sapma[j])
            .collect();
        (y, ortalama, rstd)
    }

    fn kimlikleri_denetle(&self, kimlikler: &[u32]) -> Result<(), CikarimHatasi> {
        for jeton in kimlikler {
            if *jeton as usize >= self.spec.vocab {
                return Err(CikarimHatasi::KimlikAraligi {
                    jeton: *jeton,
                    sozluk: self.spec.vocab,
                });
            }
        }
        Ok(())
    }
}

fn ortalama(degerler: &[f64]) -> f64 {
    if degerler.is_empty() {
        return f64::NAN;
    }
    degerler.iter().sum::<f64>() / degerler.len() as f64
}

/// GELU, tanh form - the trainer's own function, not an approximation of it.
fn gelu(z: f64) -> f64 {
    let ic = (2.0 / std::f64::consts::PI).sqrt() * (z + 0.044_715 * z * z * z);
    0.5 * z * (1.0 + ic.tanh())
}

/// The trainer's softmax, written the same way so the two agree bit for bit.
fn softmax(x: &[f64]) -> Vec<f64> {
    let en_buyuk = x
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| if *b > a { *b } else { a });
    let mut y: Vec<f64> = x
        .iter()
        .map(|z| {
            if z.is_finite() {
                (z - en_buyuk).exp()
            } else {
                0.0
            }
        })
        .collect();
    let mut toplam = 0.0;
    for z in &y {
        toplam += z;
    }
    for z in &mut y {
        *z /= toplam;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use lubot_egitim::kontrol::{Hassasiyet, SIHIR, SURUM};
    use lubot_egitim::INIT_STD_EMBEDDING;

    fn kucuk_spec() -> Spec {
        Spec {
            vocab: 24,
            d_model: 8,
            n_layers: 2,
            n_heads: 2,
            d_ff: 16,
            max_seq_len: 16,
        }
    }

    fn kontrol(spec: Spec, tohum: u64) -> Kontrol {
        let p = Parametreler::mup_init(spec, tohum, INIT_STD_EMBEDDING);
        let opt = lubot_egitim::Adamw::yeni(p.toplam_ogeler(), 0.01, 0.1).expect("optimizer");
        let (adim, m, v) = opt.durum();
        Kontrol {
            spec,
            parametreler: p,
            adim: 3,
            epoch: 1,
            tohum,
            sozluk_aile: "test-aile".to_string(),
            korpus_ozeti: "c".repeat(64),
            egitim_kaybi: 3.0,
            dogrulama_kaybi: Some(3.1),
            en_iyi_dogrulama: Some(2.9),
            devam_konum: 0,
            hassasiyet: Hassasiyet::F64,
            optimizer: Some(OptimizerDurumu {
                adim,
                ogrenme_orani: 0.01,
                agirlik_sonumu: 0.1,
                m: m.to_vec(),
                v: v.to_vec(),
            }),
        }
    }

    fn cikarim(spec: Spec, tohum: u64) -> Cikarim {
        Cikarim::kontrollden(kontrol(spec, tohum))
    }

    fn dizi(uzunluk: usize) -> Vec<u32> {
        (0..uzunluk).map(|i| ((i * 5 + 3) % 24) as u32).collect()
    }

    #[test]
    fn the_cached_and_full_passes_agree() {
        // Karar verici olcum: onbellekli gecis ile sifirdan yeniden islenen
        // tam gecis ayni log-olasiligi vermeli.
        let spec = kucuk_spec();
        let c = cikarim(spec, 7);
        let rapor = c.onbellek_denetimi(&dizi(10)).expect("measurement");
        assert_eq!(rapor.konum, 9);
        assert!(rapor.onbellekli.is_finite() && rapor.onbellekli < 0.0);
        assert!(
            rapor.en_buyuk_fark_onbellek < CACHE_TOLERANCE,
            "onbellekli gecis tam gecisten ayrildi: {:.3e}",
            rapor.en_buyuk_fark_onbellek
        );
    }

    #[test]
    fn this_crate_agrees_with_the_training_forward_pass() {
        // Ucuncu gorus: egitim cekirdegi ayni dizide ayni sayiyi vermeli.
        let spec = kucuk_spec();
        let c = cikarim(spec, 11);
        let rapor = c.onbellek_denetimi(&dizi(12)).expect("measurement");
        assert!(
            rapor.en_buyuk_fark_egitim < 1e-12,
            "egitim cekirdegi ile fark {:.3e}: onbellekli {:.12} tam {:.12} egitim {:.12}",
            rapor.en_buyuk_fark_egitim,
            rapor.onbellekli,
            rapor.tam_gecis,
            rapor.egitim_cekirdegi
        );
    }

    #[test]
    fn scoring_the_tail_after_one_token_matches_the_cached_pass() {
        // Baglamsiz tek jeton puanlanmaz: ilk jetonun olasiligi sozluk
        // onselidir, metnin ne kadar beklenir oldugu degil.
        let spec = kucuk_spec();
        let c = cikarim(spec, 7);
        let metin = dizi(8);
        let tek = c.puanla(&metin[..1], &metin[1..]).expect("score");
        let rapor = c.onbellek_denetimi(&metin).expect("measurement");
        assert!(
            (tek - rapor.onbellekli).abs() < 1e-12,
            "ayni is iki sayi verdi: {tek} vs {}",
            rapor.onbellekli
        );
        assert_eq!(c.puanla(&[], &metin[..1]), Err(CikarimHatasi::BosGirdi));
    }

    #[test]
    fn a_context_changes_the_score() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 13);
        let metin = dizi(6);
        let baglamsiz = c.puanla(&[1], &metin[1..]).expect("score");
        let baglamli = c.puanla(&metin[..4], &metin[4..]).expect("score");
        assert!(
            (baglamsiz - baglamli).abs() > 1e-9,
            "baglam skoru degistirmiyor: {baglamsiz} vs {baglamli}"
        );
    }

    #[test]
    fn the_context_is_used_and_not_just_counted() {
        // Ayni uzunlukta iki farkli baglam, ayni metin icin farkli puan
        // vermeli; vermiyorsa baglam okunmuyor demektir.
        let spec = kucuk_spec();
        let c = cikarim(spec, 17);
        let metin: Vec<u32> = vec![4, 9, 2];
        let bir = c.puanla(&[1, 2, 3], &metin).expect("score");
        let iki = c.puanla(&[7, 8, 9], &metin).expect("score");
        assert!((bir - iki).abs() > 1e-9, "iki baglam ayni puani verdi");
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_refused() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 5);
        assert_eq!(
            c.puanla(&[0], &[999]),
            Err(CikarimHatasi::KimlikAraligi {
                jeton: 999,
                sozluk: spec.vocab
            })
        );
        assert!(matches!(
            c.onbellek_denetimi(&[0, 999]),
            Err(CikarimHatasi::KimlikAraligi { .. })
        ));
    }

    #[test]
    fn a_window_that_does_not_fit_is_refused() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 5);
        let uzun = dizi(spec.max_seq_len + 1);
        assert_eq!(
            c.puanla(&[], &uzun),
            Err(CikarimHatasi::PencereAsimi {
                istenen: uzun.len(),
                tavan: spec.max_seq_len
            })
        );
    }

    #[test]
    fn ranking_orders_the_candidates_and_is_deterministic() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 19);
        let baglam = vec![3u32, 1, 4];
        let adaylar = vec![dizi(4), vec![9, 9, 9, 9], vec![1, 1, 2, 3]];
        let birinci = c.pasaj_sirala(&baglam, &adaylar).expect("rank");
        let ikinci = c.pasaj_sirala(&baglam, &adaylar).expect("rank");
        assert_eq!(birinci, ikinci, "ayni girdi iki farkli siralama verdi");
        for cift in birinci.windows(2) {
            let (a, b) = (&cift[0], &cift[1]);
            assert!(
                a.puan > b.puan || (a.puan == b.puan && a.sira < b.sira),
                "siralama bozuk: {a:?} sonra {b:?}"
            );
        }
        assert_eq!(birinci.len(), adaylar.len());
    }

    #[test]
    fn a_tie_is_broken_by_the_caller_s_index_not_by_chance() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 23);
        let ayni = vec![1u32, 5, 7];
        let adaylar = vec![ayni.clone(), ayni.clone(), ayni];
        let siralama = c.pasaj_sirala(&[2], &adaylar).expect("rank");
        assert_eq!(siralama[0].sira, 0);
        assert_eq!(siralama[1].sira, 1);
        assert_eq!(siralama[2].sira, 2);
        assert_eq!(siralama[0].puan, siralama[1].puan);
    }

    #[test]
    fn a_candidate_longer_than_the_window_is_scored_on_its_tail() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 29);
        let baglam = vec![1u32, 2, 3];
        let uzun: Vec<u32> = (0..spec.max_seq_len + 10)
            .map(|i| (i % 24) as u32)
            .collect();
        let siralama = c
            .pasaj_sirala(&baglam, std::slice::from_ref(&uzun))
            .expect("rank");
        let kirpilmis = c
            .puanla(
                &baglam,
                &uzun[uzun.len() - (spec.max_seq_len - baglam.len())..],
            )
            .expect("score");
        assert!(
            (siralama[0].puan - kirpilmis).abs() < 1e-12,
            "kuyruk kirpmasi tutmuyor: {} vs {kirpilmis}",
            siralama[0].puan
        );
        assert_eq!(siralama[0].jeton, spec.max_seq_len - baglam.len());
    }

    #[test]
    fn an_empty_ranking_is_refused() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 3);
        assert_eq!(c.pasaj_sirala(&[1, 2], &[]), Err(CikarimHatasi::BosAday));
        assert_eq!(c.onbellek_denetimi(&[1]), Err(CikarimHatasi::BosGirdi));
        assert_eq!(c.puanla(&[], &[]), Err(CikarimHatasi::BosGirdi));
    }

    #[test]
    fn a_checkpoint_round_trips_through_the_file_into_scoring() {
        let spec = kucuk_spec();
        let dizin = std::env::temp_dir().join("lubot-cikarim-testi");
        std::fs::create_dir_all(&dizin).expect("temp dir");
        let yol = dizin.join("c.ckpt");
        let k = kontrol(spec, 31);
        k.yaz(&yol).expect("write");
        let c = Cikarim::yukle(&yol).expect("load");
        assert_eq!(c.spec(), spec);
        assert_eq!(c.adim(), 3);
        assert_eq!(c.sozluk_aile(), "test-aile");
        assert!(c.devam_edilebilir());
        assert_eq!(c.en_iyi_dogrulama(), Some(2.9));
        let puan = c.puanla(&[1, 2, 3], &[4, 5]).expect("score");
        assert!(puan.is_finite() && puan < 0.0);
        let dogrudan = Cikarim::kontrollden(k)
            .puanla(&[1, 2, 3], &[4, 5])
            .expect("score");
        assert!(
            (puan - dogrudan).abs() < 1e-15,
            "dosyadan okunan baska bir model"
        );
        std::fs::remove_file(&yol).ok();
    }

    #[test]
    fn a_damaged_checkpoint_file_is_refused() {
        let spec = kucuk_spec();
        let dizin = std::env::temp_dir().join("lubot-cikarim-bozuk");
        std::fs::create_dir_all(&dizin).expect("temp dir");
        let yol = dizin.join("bozuk.ckpt");
        let k = kontrol(spec, 37);
        k.yaz(&yol).expect("write");
        let mut ham = std::fs::read(&yol).expect("read");
        let orta = ham.len() / 2;
        ham[orta] ^= 0x01;
        std::fs::write(&yol, &ham).expect("write");
        match Cikarim::yukle(&yol) {
            Err(CikarimHatasi::Kontrol(sebep)) => {
                assert!(
                    sebep.contains("ozet"),
                    "beklenen ozet reddi, alinan: {sebep}"
                );
            }
            digeri => panic!("bozuk dosya kabul edildi: {digeri:?}"),
        }
        std::fs::remove_file(&yol).ok();
    }

    #[test]
    fn the_surfaces_own_magic_and_version_are_the_format_s() {
        let spec = kucuk_spec();
        let c = cikarim(spec, 41);
        assert_eq!(c.spec(), spec);
        assert_eq!(c.sozluk_boyutu(), spec.vocab);
        assert_eq!(SIHIR, b"LUBOTCKPT");
        assert_eq!(SURUM, 1);
        assert_eq!(CACHE_TOLERANCE, 1e-9);
    }
}
