//! # lubot-omurga - the encoder backbone, written here
//!
//! `crates/transformer` holds a decoder-shaped sketch; `crates/egitim` holds the
//! training kernel and the component experiments (gated MLP, engram memory,
//! multi-lane residuals, routing). What the architecture notes
//! (`docs/MIMARI-TASARIM.md`) never turned into code is the *backbone the
//! decision head is supposed to sit on*: a bidirectional encoder whose
//! attention alternates between a local window and a periodic global layer,
//! whose positions are rotary, and whose blocks carry no biases. This crate is
//! that backbone.
//!
//! ## What it is not
//!
//! It is not a trained model and it does not claim to be one. `Omurga::yeni`
//! produces a deterministically initialised set of weights so the forward pass
//! can be exercised and measured; no number here has been trained, and nothing
//! in this crate reads a checkpoint from anywhere. Every parameter is produced
//! by the seeded generator in this file (K1: the shapes are written here, the weights
//! are not).
//!
//! It is also not a generator. It maps tokens to one hidden vector per token.
//! It has no output vocabulary head, no sampling, no decode loop - there is no
//! surface here that could produce text, which is the invariant the repository
//! defends with `no-generation-variant`.
//!
//! ## Why the weights are one flat buffer
//!
//! The directive this crate answers asks for something specific: checkpoints
//! must stay weight-compatible so a branch trained on one language or domain can
//! be averaged back into the trunk. Averaging two models is trivial when both
//! are one `Vec<f32>` with the same directory over it, and a minefield when they
//! are two trees of nested structs that merely look alike. So the weights are a
//! flat buffer, [`Omurga::dizin`] names the slices, and
//! [`Omurga::sekil_imzasi`] is the string two models must agree on before
//! [`Omurga::ortala`] will touch them. A merge that cannot state what made the
//! two models compatible is a merge nobody can audit.
//!
//! ## The parameter count is derived twice
//!
//! [`Omurga::param_sayisi`] reports the length of the buffer that actually
//! exists. [`Omurga::beklenen_param_sayisi`] computes the same number from the
//! configuration with a closed formula and never looks at the buffer. A test
//! asserts they agree. One of them alone is a number that can be wrong in the
//! same direction as whatever produced it.
//!
//! ## Marked, not settled
//!
//! Three choices here diverge from `training/model_spec.json` and are declared
//! rather than decided: the attention scale is `1/sqrt(d_head)` and not `1/d_k`
//! (see [`dikkat::Gqa::olcek`]); the feed-forward is gated rather than the
//! two-matrix MLP the spec counts; there are no biases. This crate is a separate
//! family and does not edit the spec. Whether the family is adopted is an
//! architectural decision with an operator's stamp on it, and the open
//! forward-pass RMS finding recorded against the `lubot-a1` family (the `Θ(1)`
//! band) has not been re-measured here.

pub mod dikkat;
pub mod hadamard;
pub mod katman;
pub mod konum;
pub mod pencere;

use dikkat::{DikkatHatasi, Gqa};
use katman::{KatmanHatasi, Norm};
use konum::{Eslesme, KonumHatasi, Rope};
use pencere::{Kapsam, PencereHatasi, Plan};

/// Why the backbone refused.
#[derive(Debug, Clone, PartialEq)]
pub enum OmurgaHatasi {
    /// A rotary position refused.
    Konum(KonumHatasi),
    /// The attention schedule refused.
    Pencere(PencereHatasi),
    /// A primitive refused.
    Katman(KatmanHatasi),
    /// Attention refused.
    DikkatKatmani(DikkatHatasi),
    /// A configuration field is zero where zero has no meaning.
    SifirAlan { alan: &'static str },
    /// The width does not divide evenly among the query heads.
    BolunmezGenislik { d_model: usize, n_kafa: usize },
    /// The directory has no tensor under that name.
    TensorYok { ad: String },
    /// A token id is outside the vocabulary.
    JetonAralikDisi { jeton: u32, vocab: usize },
    /// The sequence is longer than the configured maximum.
    DiziCokUzun { dizi: usize, azami: usize },
    /// An empty sequence has no hidden states.
    BosDizi,
    /// Two models were asked to merge without the same shape.
    UyumsuzImza { sol: String, sag: String },
    /// An average of no models is not a model.
    BosTopluluk,
}

impl std::fmt::Display for OmurgaHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Konum(e) => write!(f, "rotary positions: {e}"),
            Self::Pencere(e) => write!(f, "attention schedule: {e}"),
            Self::Katman(e) => write!(f, "layer primitive: {e}"),
            Self::DikkatKatmani(e) => write!(f, "attention: {e}"),
            Self::SifirAlan { alan } => write!(f, "the configuration field `{alan}` is zero"),
            Self::BolunmezGenislik { d_model, n_kafa } => write!(
                f,
                "a width of {d_model} does not divide evenly among {n_kafa} query heads"
            ),
            Self::TensorYok { ad } => write!(f, "there is no tensor named `{ad}`"),
            Self::JetonAralikDisi { jeton, vocab } => write!(
                f,
                "token {jeton} is outside a vocabulary of {vocab}"
            ),
            Self::DiziCokUzun { dizi, azami } => write!(
                f,
                "a sequence of {dizi} is longer than the configured maximum of {azami}"
            ),
            Self::BosDizi => write!(f, "an empty sequence has no hidden states"),
            Self::UyumsuzImza { sol, sag } => write!(
                f,
                "these models do not share a shape, so averaging them is undefined: `{sol}` against `{sag}`"
            ),
            Self::BosTopluluk => write!(f, "an average of no models is not a model"),
        }
    }
}

impl From<KonumHatasi> for OmurgaHatasi {
    fn from(e: KonumHatasi) -> Self {
        Self::Konum(e)
    }
}

impl From<PencereHatasi> for OmurgaHatasi {
    fn from(e: PencereHatasi) -> Self {
        Self::Pencere(e)
    }
}

impl From<KatmanHatasi> for OmurgaHatasi {
    fn from(e: KatmanHatasi) -> Self {
        Self::Katman(e)
    }
}

impl From<DikkatHatasi> for OmurgaHatasi {
    fn from(e: DikkatHatasi) -> Self {
        Self::DikkatKatmani(e)
    }
}

/// The shape of the backbone.
#[derive(Debug, Clone, PartialEq)]
pub struct Yapilandirma {
    pub d_model: usize,
    pub n_katman: usize,
    pub n_sorgu_kafa: usize,
    pub n_kv_kafa: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub azami_dizi: usize,
    /// One layer in this many is global.
    pub genel_periyot: usize,
    /// The half-width of the sliding window on the other layers.
    pub yerel_yaricap: usize,
    pub rope_taban: f64,
    pub eslesme: Eslesme,
    pub norm_eps: f32,
    /// The width-independent initialisation scale for the embedding, declared
    /// rather than measured. The hidden tensors follow `sqrt(2/fan_in)`, which
    /// is the rule `training/model_spec.json` already declares for the `a1`
    /// family; this is the one value that rule does not fix.
    pub gomme_init_std: f32,
}

impl Yapilandirma {
    /// A small candidate sized against the frozen vocabulary family this
    /// repository already carries (8192), not against any published model.
    ///
    /// Nothing about these numbers has been measured to be good. They are small
    /// enough that a test can run a forward pass through the whole stack, which
    /// is the only property claimed for them.
    #[must_use]
    pub fn kucuk_aday() -> Self {
        Self {
            d_model: 128,
            n_katman: 6,
            n_sorgu_kafa: 4,
            n_kv_kafa: 2,
            d_ff: 256,
            vocab: 8192,
            azami_dizi: 256,
            genel_periyot: 3,
            yerel_yaricap: 32,
            rope_taban: 10_000.0,
            eslesme: Eslesme::YariyaBolme,
            norm_eps: 1e-5,
            gomme_init_std: 0.02,
        }
    }

    /// The per-head width.
    ///
    /// # Errors
    ///
    /// [`OmurgaHatasi::SifirAlan`] or [`OmurgaHatasi::BolunmezGenislik`].
    pub fn d_head(&self) -> Result<usize, OmurgaHatasi> {
        if self.n_sorgu_kafa == 0 {
            return Err(OmurgaHatasi::SifirAlan {
                alan: "n_sorgu_kafa",
            });
        }
        if self.d_model == 0 {
            return Err(OmurgaHatasi::SifirAlan { alan: "d_model" });
        }
        if !self.d_model.is_multiple_of(self.n_sorgu_kafa) {
            return Err(OmurgaHatasi::BolunmezGenislik {
                d_model: self.d_model,
                n_kafa: self.n_sorgu_kafa,
            });
        }
        Ok(self.d_model / self.n_sorgu_kafa)
    }

    /// Checks every field that has no meaningful zero.
    ///
    /// # Errors
    ///
    /// [`OmurgaHatasi::SifirAlan`], plus whatever [`Yapilandirma::d_head`]
    /// refuses.
    pub fn dogrula(&self) -> Result<(), OmurgaHatasi> {
        for (deger, alan) in [
            (self.n_katman, "n_katman"),
            (self.d_ff, "d_ff"),
            (self.vocab, "vocab"),
            (self.azami_dizi, "azami_dizi"),
        ] {
            if deger == 0 {
                return Err(OmurgaHatasi::SifirAlan { alan });
            }
        }
        self.d_head()?;
        Ok(())
    }
}

/// One named slice of the weight buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorKaydi {
    pub ad: String,
    pub ofset: usize,
    pub satir: usize,
    pub sutun: usize,
}

impl TensorKaydi {
    #[must_use]
    pub fn uzunluk(&self) -> usize {
        self.satir * self.sutun
    }
}

/// A reproducible generator.
///
/// xorshift64*, then Box-Muller. Not cryptographic and not meant to be: the
/// requirement is that two machines given the same seed produce the same
/// weights, which is what makes a checkpoint comparison meaningful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tohum {
    durum: u64,
}

impl Tohum {
    /// A zero seed is mapped away, because xorshift is stuck at zero forever.
    #[must_use]
    pub fn yeni(tohum: u64) -> Self {
        Self {
            durum: if tohum == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                tohum
            },
        }
    }

    fn sonraki(&mut self) -> u64 {
        let mut x = self.durum;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.durum = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on `(0, 1)`; never exactly zero, so the logarithm is safe.
    fn birim(&mut self) -> f64 {
        let ham = self.sonraki() >> 11;
        ((ham as f64) + 0.5) / ((1u64 << 53) as f64)
    }

    /// One raw draw, for callers that need a whole number rather than a
    /// normal sample - a frozen permutation, for instance, where a Gaussian
    /// would have to be rounded and the rounding would bias the shuffle.
    #[must_use]
    pub fn tam_sayi(&mut self) -> u64 {
        self.sonraki()
    }

    /// One standard normal sample.
    #[must_use]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.birim();
        let u2 = self.birim();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The backbone.
#[derive(Debug, Clone, PartialEq)]
pub struct Omurga {
    yap: Yapilandirma,
    plan: Plan,
    rope: Rope,
    gqa: Gqa,
    norm: Norm,
    dizin: Vec<TensorKaydi>,
    agirlik: Vec<f32>,
}

/// The tensor names, in the order they occupy the buffer.
fn tensor_plani(yap: &Yapilandirma) -> Result<Vec<TensorKaydi>, OmurgaHatasi> {
    let d_head = yap.d_head()?;
    let q_genislik = yap.n_sorgu_kafa * d_head;
    let kv_genislik = yap.n_kv_kafa * d_head;
    let mut dizin: Vec<TensorKaydi> = Vec::new();
    let mut ofset = 0usize;
    let mut ekle = |ad: String, satir: usize, sutun: usize, ofset: &mut usize| {
        dizin.push(TensorKaydi {
            ad,
            ofset: *ofset,
            satir,
            sutun,
        });
        *ofset += satir * sutun;
    };
    ekle("gomme".to_string(), yap.vocab, yap.d_model, &mut ofset);
    for katman in 0..yap.n_katman {
        ekle(
            format!("katman.{katman}.dikkat_norm"),
            1,
            yap.d_model,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.wq"),
            q_genislik,
            yap.d_model,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.wk"),
            kv_genislik,
            yap.d_model,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.wv"),
            kv_genislik,
            yap.d_model,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.wo"),
            yap.d_model,
            q_genislik,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.mlp_norm"),
            1,
            yap.d_model,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.w_giris"),
            2 * yap.d_ff,
            yap.d_model,
            &mut ofset,
        );
        ekle(
            format!("katman.{katman}.w_cikis"),
            yap.d_model,
            yap.d_ff,
            &mut ofset,
        );
    }
    ekle("son_norm".to_string(), 1, yap.d_model, &mut ofset);
    Ok(dizin)
}

/// Whether a tensor is a normalisation gain (initialised to one) rather than a
/// projection (initialised from the generator).
fn kazanc_mi(ad: &str) -> bool {
    ad.ends_with("_norm")
}

impl Omurga {
    /// Builds a deterministically initialised backbone.
    ///
    /// # Errors
    ///
    /// Whatever [`Yapilandirma::dogrula`], the schedule, the rotary map or the
    /// head layout refuses.
    pub fn yeni(yap: Yapilandirma, tohum: u64) -> Result<Self, OmurgaHatasi> {
        yap.dogrula()?;
        let d_head = yap.d_head()?;
        let plan = Plan::periyodik(yap.n_katman, yap.genel_periyot, yap.yerel_yaricap)?;
        let rope = Rope::yeni(d_head, yap.rope_taban, yap.eslesme)?;
        let gqa = Gqa::yeni(yap.n_sorgu_kafa, yap.n_kv_kafa, d_head)?;
        let norm = Norm::yeni(yap.d_model, yap.norm_eps)?;
        let dizin = tensor_plani(&yap)?;
        let toplam: usize = dizin.iter().map(TensorKaydi::uzunluk).sum();
        let mut agirlik = vec![0.0f32; toplam];
        let mut uretec = Tohum::yeni(tohum);
        for kayit in &dizin {
            let dilim = &mut agirlik[kayit.ofset..kayit.ofset + kayit.uzunluk()];
            if kazanc_mi(&kayit.ad) {
                for v in dilim.iter_mut() {
                    *v = 1.0;
                }
                continue;
            }
            // `sqrt(2/fan_in)` for the projections, the width-independent scale
            // for the embedding: the split `training/model_spec.json` already
            // declares for the other family.
            let std = if kayit.ad == "gomme" {
                f64::from(yap.gomme_init_std)
            } else {
                (2.0 / (kayit.sutun as f64)).sqrt()
            };
            for v in dilim.iter_mut() {
                *v = (uretec.normal() * std) as f32;
            }
        }
        Ok(Self {
            yap,
            plan,
            rope,
            gqa,
            norm,
            dizin,
            agirlik,
        })
    }

    #[must_use]
    pub fn yapilandirma(&self) -> &Yapilandirma {
        &self.yap
    }

    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    #[must_use]
    pub fn rope(&self) -> &Rope {
        &self.rope
    }

    #[must_use]
    pub fn gqa(&self) -> &Gqa {
        &self.gqa
    }

    #[must_use]
    pub fn dizin(&self) -> &[TensorKaydi] {
        &self.dizin
    }

    #[must_use]
    pub fn agirlik(&self) -> &[f32] {
        &self.agirlik
    }

    /// The number of parameters this model actually holds.
    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        self.agirlik.len()
    }

    /// The same number from a closed formula, without looking at the buffer.
    ///
    /// # Errors
    ///
    /// Whatever [`Yapilandirma::d_head`] refuses.
    pub fn beklenen_param_sayisi(yap: &Yapilandirma) -> Result<usize, OmurgaHatasi> {
        let d_head = yap.d_head()?;
        let q = yap.n_sorgu_kafa * d_head;
        let kv = yap.n_kv_kafa * d_head;
        let gomme = yap.vocab * yap.d_model;
        let katman_basi = 2 * yap.d_model              // the two gains
            + q * yap.d_model                          // wq
            + 2 * kv * yap.d_model                     // wk, wv
            + yap.d_model * q                          // wo
            + 2 * yap.d_ff * yap.d_model               // the gated up-projection
            + yap.d_model * yap.d_ff; // the down-projection
        Ok(gomme + yap.n_katman * katman_basi + yap.d_model)
    }

    /// A named slice of the weight buffer.
    ///
    /// # Errors
    ///
    /// [`OmurgaHatasi::TensorYok`].
    pub fn tensor(&self, ad: &str) -> Result<&[f32], OmurgaHatasi> {
        let kayit = self
            .dizin
            .iter()
            .find(|k| k.ad == ad)
            .ok_or_else(|| OmurgaHatasi::TensorYok { ad: ad.to_string() })?;
        Ok(&self.agirlik[kayit.ofset..kayit.ofset + kayit.uzunluk()])
    }

    /// The string two models must share before they can be averaged.
    ///
    /// Everything that changes a tensor shape is in it, and nothing that does
    /// not. The seed is deliberately absent: two models trained from different
    /// seeds are exactly the case merging exists for.
    #[must_use]
    pub fn sekil_imzasi(&self) -> String {
        format!(
            "omurga-1|d={}|L={}|qh={}|kvh={}|dff={}|V={}|per={}|r={}|rope={}|taban={}",
            self.yap.d_model,
            self.yap.n_katman,
            self.yap.n_sorgu_kafa,
            self.yap.n_kv_kafa,
            self.yap.d_ff,
            self.yap.vocab,
            self.yap.genel_periyot,
            self.yap.yerel_yaricap,
            self.yap.eslesme.ad(),
            self.yap.rope_taban,
        )
    }

    /// Whether two models may be merged.
    #[must_use]
    pub fn imza_uyumlu(&self, diger: &Self) -> bool {
        self.sekil_imzasi() == diger.sekil_imzasi()
    }

    /// Applies rotary positions to every head of a projected sequence in place.
    fn konumla(&self, x: &mut [f32], kafa_sayisi: usize) -> Result<(), OmurgaHatasi> {
        let d_head = self.rope.d_head();
        let satir = kafa_sayisi * d_head;
        for (pos, dilim) in x.chunks_exact_mut(satir).enumerate() {
            for kafa in dilim.chunks_exact_mut(d_head) {
                self.rope.uygula(kafa, pos)?;
            }
        }
        Ok(())
    }

    /// One hidden vector per token: `dizi` rows of `d_model`.
    ///
    /// # Errors
    ///
    /// [`OmurgaHatasi::BosDizi`], [`OmurgaHatasi::DiziCokUzun`],
    /// [`OmurgaHatasi::JetonAralikDisi`], or whatever a primitive refuses.
    pub fn ileri(&self, jetonlar: &[u32]) -> Result<Vec<f32>, OmurgaHatasi> {
        if jetonlar.is_empty() {
            return Err(OmurgaHatasi::BosDizi);
        }
        if jetonlar.len() > self.yap.azami_dizi {
            return Err(OmurgaHatasi::DiziCokUzun {
                dizi: jetonlar.len(),
                azami: self.yap.azami_dizi,
            });
        }
        let d_model = self.yap.d_model;
        let d_head = self.rope.d_head();
        let q_genislik = self.yap.n_sorgu_kafa * d_head;
        let kv_genislik = self.yap.n_kv_kafa * d_head;
        let gomme = self.tensor("gomme")?;
        let mut x = Vec::with_capacity(jetonlar.len() * d_model);
        for jeton in jetonlar {
            let idx = *jeton as usize;
            if idx >= self.yap.vocab {
                return Err(OmurgaHatasi::JetonAralikDisi {
                    jeton: *jeton,
                    vocab: self.yap.vocab,
                });
            }
            x.extend_from_slice(&gomme[idx * d_model..(idx + 1) * d_model]);
        }

        for katman in 0..self.yap.n_katman {
            let kapsam: Kapsam = self.plan.kapsam(katman)?;
            let gor = move |sorgu: usize, anahtar: usize| kapsam.gorulebilir(sorgu, anahtar);

            let h = self
                .norm
                .uygula(&x, self.tensor(&format!("katman.{katman}.dikkat_norm"))?)?;
            let mut q = katman::carp(
                self.tensor(&format!("katman.{katman}.wq"))?,
                q_genislik,
                d_model,
                &h,
            )?;
            let mut k = katman::carp(
                self.tensor(&format!("katman.{katman}.wk"))?,
                kv_genislik,
                d_model,
                &h,
            )?;
            let v = katman::carp(
                self.tensor(&format!("katman.{katman}.wv"))?,
                kv_genislik,
                d_model,
                &h,
            )?;
            self.konumla(&mut q, self.yap.n_sorgu_kafa)?;
            self.konumla(&mut k, self.yap.n_kv_kafa)?;
            let dikkat = self.gqa.ileri(&q, &k, &v, &gor)?;
            let cikti = katman::carp(
                self.tensor(&format!("katman.{katman}.wo"))?,
                d_model,
                q_genislik,
                &dikkat,
            )?;
            for (yuva, deger) in x.iter_mut().zip(cikti.iter()) {
                *yuva += deger;
            }

            let h2 = self
                .norm
                .uygula(&x, self.tensor(&format!("katman.{katman}.mlp_norm"))?)?;
            let mlp = katman::kapili_ileri(
                self.tensor(&format!("katman.{katman}.w_giris"))?,
                self.tensor(&format!("katman.{katman}.w_cikis"))?,
                d_model,
                self.yap.d_ff,
                &h2,
            )?;
            for (yuva, deger) in x.iter_mut().zip(mlp.iter()) {
                *yuva += deger;
            }
        }

        Ok(self.norm.uygula(&x, self.tensor("son_norm")?)?)
    }

    /// The root-mean-square of a hidden state, per position.
    ///
    /// The `lubot-a1` family has an open finding about the forward-pass RMS
    /// profile leaving its band. This is the instrument that would let the same
    /// question be asked of this family; it answers it for nobody here.
    #[must_use]
    pub fn konum_rms(durum: &[f32], d_model: usize) -> Vec<f32> {
        if d_model == 0 {
            return Vec::new();
        }
        durum
            .chunks_exact(d_model)
            .map(|satir| {
                let kare: f64 = satir.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
                (kare / (d_model as f64)).sqrt() as f32
            })
            .collect()
    }

    /// The parameter-space average of several models with the same shape.
    ///
    /// # Errors
    ///
    /// [`OmurgaHatasi::BosTopluluk`] or [`OmurgaHatasi::UyumsuzImza`].
    pub fn ortala(modeller: &[&Self]) -> Result<Self, OmurgaHatasi> {
        let Some((ilk, kalan)) = modeller.split_first() else {
            return Err(OmurgaHatasi::BosTopluluk);
        };
        for diger in kalan {
            if !ilk.imza_uyumlu(diger) {
                return Err(OmurgaHatasi::UyumsuzImza {
                    sol: ilk.sekil_imzasi(),
                    sag: diger.sekil_imzasi(),
                });
            }
        }
        let n = modeller.len() as f64;
        let mut toplam = vec![0.0f64; ilk.agirlik.len()];
        for model in modeller {
            for (yuva, deger) in toplam.iter_mut().zip(model.agirlik.iter()) {
                *yuva += f64::from(*deger);
            }
        }
        let agirlik = toplam.iter().map(|v| (*v / n) as f32).collect();
        Ok(Self {
            yap: ilk.yap.clone(),
            plan: ilk.plan.clone(),
            rope: ilk.rope.clone(),
            gqa: ilk.gqa,
            norm: ilk.norm.clone(),
            dizin: ilk.dizin.clone(),
            agirlik,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kucuk() -> Yapilandirma {
        Yapilandirma {
            d_model: 16,
            n_katman: 4,
            n_sorgu_kafa: 4,
            n_kv_kafa: 2,
            d_ff: 24,
            vocab: 64,
            azami_dizi: 32,
            genel_periyot: 3,
            yerel_yaricap: 2,
            rope_taban: 10_000.0,
            eslesme: Eslesme::YariyaBolme,
            norm_eps: 1e-5,
            gomme_init_std: 0.02,
        }
    }

    #[test]
    fn aday_yapilandirma_gecerli() {
        Yapilandirma::kucuk_aday().dogrula().unwrap();
    }

    #[test]
    fn sifir_alan_reddedilir() {
        let mut yap = kucuk();
        yap.n_katman = 0;
        assert_eq!(
            yap.dogrula(),
            Err(OmurgaHatasi::SifirAlan { alan: "n_katman" })
        );
        let mut yap = kucuk();
        yap.d_ff = 0;
        assert_eq!(yap.dogrula(), Err(OmurgaHatasi::SifirAlan { alan: "d_ff" }));
        let mut yap = kucuk();
        yap.vocab = 0;
        assert_eq!(
            yap.dogrula(),
            Err(OmurgaHatasi::SifirAlan { alan: "vocab" })
        );
        let mut yap = kucuk();
        yap.azami_dizi = 0;
        assert_eq!(
            yap.dogrula(),
            Err(OmurgaHatasi::SifirAlan { alan: "azami_dizi" })
        );
        let mut yap = kucuk();
        yap.d_model = 0;
        assert_eq!(
            yap.d_head(),
            Err(OmurgaHatasi::SifirAlan { alan: "d_model" })
        );
        let mut yap = kucuk();
        yap.n_sorgu_kafa = 0;
        assert_eq!(
            yap.d_head(),
            Err(OmurgaHatasi::SifirAlan {
                alan: "n_sorgu_kafa"
            })
        );
    }

    #[test]
    fn bolunmez_genislik_reddedilir() {
        let mut yap = kucuk();
        yap.d_model = 18;
        assert_eq!(
            yap.d_head(),
            Err(OmurgaHatasi::BolunmezGenislik {
                d_model: 18,
                n_kafa: 4
            })
        );
    }

    #[test]
    fn param_sayisi_iki_yoldan_ayni() {
        let yap = kucuk();
        let model = Omurga::yeni(yap.clone(), 7).unwrap();
        assert_eq!(
            model.param_sayisi(),
            Omurga::beklenen_param_sayisi(&yap).unwrap(),
            "the directory and the closed formula disagree"
        );
    }

    #[test]
    fn param_sayisi_aday_icin_de_ayni() {
        let yap = Yapilandirma::kucuk_aday();
        let model = Omurga::yeni(yap.clone(), 1).unwrap();
        assert_eq!(
            model.param_sayisi(),
            Omurga::beklenen_param_sayisi(&yap).unwrap()
        );
    }

    #[test]
    fn dizin_bosluksuz_ve_ortusmez() {
        let model = Omurga::yeni(kucuk(), 3).unwrap();
        let mut beklenen = 0usize;
        for kayit in model.dizin() {
            assert_eq!(
                kayit.ofset, beklenen,
                "a hole or an overlap at {}",
                kayit.ad
            );
            beklenen += kayit.uzunluk();
        }
        assert_eq!(beklenen, model.param_sayisi());
    }

    #[test]
    fn dizin_adlari_benzersiz() {
        let model = Omurga::yeni(kucuk(), 3).unwrap();
        let mut adlar: Vec<&str> = model.dizin().iter().map(|k| k.ad.as_str()).collect();
        let toplam = adlar.len();
        adlar.sort_unstable();
        adlar.dedup();
        assert_eq!(adlar.len(), toplam, "two tensors share a name");
    }

    #[test]
    fn normlar_bire_kurulur() {
        let model = Omurga::yeni(kucuk(), 3).unwrap();
        for ad in ["katman.0.dikkat_norm", "katman.2.mlp_norm", "son_norm"] {
            let t = model.tensor(ad).unwrap();
            assert!(
                t.iter().all(|v| (*v - 1.0).abs() < 1e-9),
                "{ad} was not initialised to one"
            );
        }
    }

    #[test]
    fn projeksiyonlar_sifir_degil() {
        let model = Omurga::yeni(kucuk(), 3).unwrap();
        let wq = model.tensor("katman.0.wq").unwrap();
        assert!(wq.iter().any(|v| v.abs() > 1e-6), "wq is all zeros");
    }

    #[test]
    fn init_olcegi_fan_in_kuralini_izler() {
        // sqrt(2/fan_in) with fan_in = d_model = 16 gives 0.3536. A sample
        // standard deviation is noisy, so the band is wide; what is being
        // checked is that the rule was applied at all, not its third digit.
        let model = Omurga::yeni(kucuk(), 11).unwrap();
        let wq = model.tensor("katman.0.wq").unwrap();
        let kare: f64 = wq.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let olculen = (kare / (wq.len() as f64)).sqrt();
        let beklenen = (2.0f64 / 16.0).sqrt();
        assert!(
            (olculen / beklenen - 1.0).abs() < 0.25,
            "measured {olculen}, the rule asks for about {beklenen}"
        );
    }

    #[test]
    fn tensor_yok_reddedilir() {
        let model = Omurga::yeni(kucuk(), 3).unwrap();
        assert_eq!(
            model.tensor("yok-boyle-bir-sey"),
            Err(OmurgaHatasi::TensorYok {
                ad: "yok-boyle-bir-sey".to_string()
            })
        );
    }

    #[test]
    fn ayni_tohum_ayni_agirlik() {
        let a = Omurga::yeni(kucuk(), 42).unwrap();
        let b = Omurga::yeni(kucuk(), 42).unwrap();
        assert_eq!(a.agirlik(), b.agirlik());
    }

    #[test]
    fn farkli_tohum_farkli_agirlik() {
        let a = Omurga::yeni(kucuk(), 42).unwrap();
        let b = Omurga::yeni(kucuk(), 43).unwrap();
        assert_ne!(a.agirlik(), b.agirlik());
    }

    #[test]
    fn sifir_tohum_takilmaz() {
        let model = Omurga::yeni(kucuk(), 0).unwrap();
        let wq = model.tensor("katman.0.wq").unwrap();
        assert!(
            wq.iter().any(|v| v.abs() > 1e-6),
            "a zero seed produced a dead generator"
        );
    }

    #[test]
    fn uretec_makul_bir_normal_verir() {
        let mut uretec = Tohum::yeni(9);
        let n = 4000;
        let ornekler: Vec<f64> = (0..n).map(|_| uretec.normal()).collect();
        let ortalama: f64 = ornekler.iter().sum::<f64>() / f64::from(n);
        let varyans: f64 = ornekler
            .iter()
            .map(|v| (v - ortalama) * (v - ortalama))
            .sum::<f64>()
            / f64::from(n);
        assert!(ortalama.abs() < 0.1, "mean drifted to {ortalama}");
        assert!(
            (varyans - 1.0).abs() < 0.15,
            "variance is {varyans}, not about one"
        );
    }

    #[test]
    fn ileri_gecis_sekli_dogru() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        let jetonlar = [1u32, 2, 3, 4, 5, 6, 7];
        let durum = model.ileri(&jetonlar).unwrap();
        assert_eq!(durum.len(), jetonlar.len() * 16);
    }

    #[test]
    fn ileri_gecis_sonlu() {
        let model = Omurga::yeni(Yapilandirma::kucuk_aday(), 2).unwrap();
        let jetonlar: Vec<u32> = (0..48).collect();
        let durum = model.ileri(&jetonlar).unwrap();
        assert!(
            durum.iter().all(|v| v.is_finite()),
            "the forward pass produced a non-finite value"
        );
    }

    #[test]
    fn ileri_gecis_deterministik() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        let jetonlar = [3u32, 1, 4, 1, 5, 9];
        assert_eq!(
            model.ileri(&jetonlar).unwrap(),
            model.ileri(&jetonlar).unwrap()
        );
    }

    #[test]
    fn ileri_gecis_jetona_duyarli() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        let a = model.ileri(&[1, 2, 3, 4]).unwrap();
        let b = model.ileri(&[1, 2, 3, 5]).unwrap();
        assert_ne!(a, b, "changing a token changed nothing");
    }

    #[test]
    fn ileri_gecis_konuma_duyarli() {
        // Rotary positions are the only thing distinguishing these two, so if
        // they come out equal the positions are not reaching the scores.
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        let a = model.ileri(&[7, 9]).unwrap();
        let b = model.ileri(&[9, 7]).unwrap();
        let ilk_a = &a[..16];
        let ikinci_b = &b[16..];
        let fark: f32 = ilk_a
            .iter()
            .zip(ikinci_b.iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max);
        assert!(fark > 1e-5, "position made no difference: {fark}");
    }

    #[test]
    fn yerel_pencere_uzaktaki_jetonu_ilk_katmanda_gecirmez() {
        // Layer 0 is global, so a single-layer model must see everything; a
        // model whose only layer is local must not. This is the schedule being
        // load-bearing rather than decorative.
        let mut yap = kucuk();
        yap.n_katman = 1;
        yap.genel_periyot = 3;
        let genel = Omurga::yeni(yap.clone(), 5).unwrap();
        assert_eq!(genel.plan().kapsam(0), Ok(Kapsam::Genel));

        let uzun: Vec<u32> = (0..20).collect();
        let mut degisik = uzun.clone();
        degisik[19] = 33;
        let a = genel.ileri(&uzun).unwrap();
        let b = genel.ileri(&degisik).unwrap();
        let ilk_fark: f32 = a[..16]
            .iter()
            .zip(b[..16].iter())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max);
        assert!(
            ilk_fark > 1e-6,
            "a global layer did not carry a distant token to position zero"
        );
    }

    #[test]
    fn bos_dizi_reddedilir() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        assert_eq!(model.ileri(&[]), Err(OmurgaHatasi::BosDizi));
    }

    #[test]
    fn uzun_dizi_reddedilir() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        let uzun: Vec<u32> = (0..33).map(|i| i % 64).collect();
        assert_eq!(
            model.ileri(&uzun),
            Err(OmurgaHatasi::DiziCokUzun {
                dizi: 33,
                azami: 32
            })
        );
    }

    #[test]
    fn aralik_disi_jeton_reddedilir() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        assert_eq!(
            model.ileri(&[0, 64]),
            Err(OmurgaHatasi::JetonAralikDisi {
                jeton: 64,
                vocab: 64
            })
        );
    }

    #[test]
    fn konum_rms_konum_basina_bir_sayi() {
        let model = Omurga::yeni(kucuk(), 5).unwrap();
        let durum = model.ileri(&[1, 2, 3]).unwrap();
        let profil = Omurga::konum_rms(&durum, 16);
        assert_eq!(profil.len(), 3);
        assert!(profil.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(Omurga::konum_rms(&durum, 0).is_empty());
    }

    #[test]
    fn imza_sekli_tasir_tohumu_tasimaz() {
        let a = Omurga::yeni(kucuk(), 1).unwrap();
        let b = Omurga::yeni(kucuk(), 2).unwrap();
        assert_eq!(a.sekil_imzasi(), b.sekil_imzasi());
        assert!(a.imza_uyumlu(&b));

        let mut baska = kucuk();
        baska.d_ff = 32;
        let c = Omurga::yeni(baska, 1).unwrap();
        assert_ne!(a.sekil_imzasi(), c.sekil_imzasi());
        assert!(!a.imza_uyumlu(&c));
    }

    #[test]
    fn imza_eslesmeyi_ayirir() {
        let a = Omurga::yeni(kucuk(), 1).unwrap();
        let mut baska = kucuk();
        baska.eslesme = Eslesme::KomsuCift;
        let b = Omurga::yeni(baska, 1).unwrap();
        assert!(
            !a.imza_uyumlu(&b),
            "two different rotary pairings were treated as mergeable"
        );
    }

    #[test]
    fn kendisiyle_ortalama_ayni_model() {
        let a = Omurga::yeni(kucuk(), 13).unwrap();
        let birlesik = Omurga::ortala(&[&a, &a]).unwrap();
        for (p, q) in a.agirlik().iter().zip(birlesik.agirlik().iter()) {
            assert!((p - q).abs() < 1e-6);
        }
        assert_eq!(
            a.ileri(&[1, 2, 3]).unwrap().len(),
            birlesik.ileri(&[1, 2, 3]).unwrap().len()
        );
    }

    #[test]
    fn ortalama_gercekten_ortalar() {
        let a = Omurga::yeni(kucuk(), 13).unwrap();
        let b = Omurga::yeni(kucuk(), 14).unwrap();
        let birlesik = Omurga::ortala(&[&a, &b]).unwrap();
        for ((p, q), r) in a
            .agirlik()
            .iter()
            .zip(b.agirlik().iter())
            .zip(birlesik.agirlik().iter())
        {
            assert!(((p + q) / 2.0 - r).abs() < 1e-6);
        }
    }

    #[test]
    fn ortalama_calisan_bir_model_verir() {
        let a = Omurga::yeni(kucuk(), 13).unwrap();
        let b = Omurga::yeni(kucuk(), 14).unwrap();
        let birlesik = Omurga::ortala(&[&a, &b]).unwrap();
        let durum = birlesik.ileri(&[2, 4, 6, 8]).unwrap();
        assert_eq!(durum.len(), 4 * 16);
        assert!(durum.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn uyumsuz_imza_birlesmez() {
        let a = Omurga::yeni(kucuk(), 13).unwrap();
        let mut baska = kucuk();
        baska.n_katman = 5;
        let b = Omurga::yeni(baska, 13).unwrap();
        match Omurga::ortala(&[&a, &b]) {
            Err(OmurgaHatasi::UyumsuzImza { sol, sag }) => assert_ne!(sol, sag),
            other => panic!("a shape mismatch was merged anyway: {other:?}"),
        }
    }

    #[test]
    fn bos_topluluk_reddedilir() {
        assert_eq!(Omurga::ortala(&[]), Err(OmurgaHatasi::BosTopluluk));
    }

    #[test]
    fn plan_ve_kafa_duzeni_geri_okunur() {
        let model = Omurga::yeni(kucuk(), 1).unwrap();
        assert_eq!(model.plan().ozet(), "GyyG");
        assert_eq!(model.gqa().grup_boyutu(), 2);
        assert_eq!(model.rope().eslesme(), Eslesme::YariyaBolme);
        assert_eq!(model.rope().olculen_eslesme(), Eslesme::YariyaBolme);
        assert_eq!(model.yapilandirma().d_model, 16);
    }

    #[test]
    fn hata_metinleri_baglami_tasir() {
        assert!(OmurgaHatasi::from(KonumHatasi::SifirBoyut)
            .to_string()
            .contains("rotary"));
        assert!(OmurgaHatasi::from(PencereHatasi::SifirKatman)
            .to_string()
            .contains("schedule"));
        assert!(OmurgaHatasi::from(KatmanHatasi::SifirGenislik)
            .to_string()
            .contains("primitive"));
        assert!(OmurgaHatasi::from(DikkatHatasi::SifirKafa)
            .to_string()
            .contains("attention"));
        assert!(OmurgaHatasi::SifirAlan { alan: "d_ff" }
            .to_string()
            .contains("d_ff"));
        assert!(OmurgaHatasi::BolunmezGenislik {
            d_model: 18,
            n_kafa: 4
        }
        .to_string()
        .contains("18"));
        assert!(OmurgaHatasi::TensorYok { ad: "a".into() }
            .to_string()
            .contains('a'));
        assert!(OmurgaHatasi::JetonAralikDisi {
            jeton: 64,
            vocab: 64
        }
        .to_string()
        .contains("64"));
        assert!(OmurgaHatasi::DiziCokUzun {
            dizi: 33,
            azami: 32
        }
        .to_string()
        .contains("33"));
        assert!(!OmurgaHatasi::BosDizi.to_string().is_empty());
        assert!(OmurgaHatasi::UyumsuzImza {
            sol: "a".into(),
            sag: "b".into()
        }
        .to_string()
        .contains('b'));
        assert!(!OmurgaHatasi::BosTopluluk.to_string().is_empty());
    }

    #[test]
    fn tensor_kaydi_uzunlugu() {
        let kayit = TensorKaydi {
            ad: "x".to_string(),
            ofset: 0,
            satir: 3,
            sutun: 4,
        };
        assert_eq!(kayit.uzunluk(), 12);
    }
}
