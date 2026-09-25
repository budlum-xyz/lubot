//! The encoder: attention, the gated feed-forward, and the layer stack.
//!
//! # What one layer does, in order
//!
//! ```text
//! x = x + attn(norm(x))     # attention attends, residual carries
//! x = x + mlp(norm(x))      # feed-forward widens, residual carries
//! ```
//!
//! The order matters and is not a convention: normalising *before* the branch
//! is what makes a deep stack trainable without warm-up, and the residual on
//! the outside is what keeps the signal from the embedding reachable. A port
//! that normalised after the addition would produce a model that looks right
//! for one layer and degrades with depth.
//!
//! # The gated feed-forward
//!
//! The stored `Wi` is `[2 * ffn, hidden]`: the first `ffn` rows are the gate,
//! the second `ffn` rows are the value, and the layer returns
//! `Wo @ (gelu(gate) * value)`. The split point is `ffn = Wi.len() / (2 *
//! hidden)`, derived from the tensor itself rather than from the configuration,
//! so a checkpoint whose feed-forward width differs still runs - and the test
//! below pins which half is which, because swapping them is silent.
//!
//! # Sliding attention
//!
//! A sliding layer attends only to the `w` positions ending at the query.
//! Implementing it as a mask over the scores (rather than as a loop over
//! windows) keeps one code path for both kinds, which is what makes the
//! comparison between them meaningful: two code paths could differ for a reason
//! that has nothing to do with attention.

use crate::baslik::{BaslikHatasi, Dizin, ParcaliDosya};
use crate::hesap::{gelu, katman_norm, matris_vektor, rope, rope_tablosu, softmax};
use crate::yapilandirma::{KatmanTuru, KodlayiciYapisi};

/// One layer's attention weights.
#[derive(Debug, Clone)]
pub struct DikkatAgirliklari {
    /// `[3 * hidden, hidden]`: query, key and value stacked in that order.
    pub wqkv: Vec<f32>,
    /// `[hidden, hidden]`.
    pub wo: Vec<f32>,
}

/// One layer's feed-forward weights.
#[derive(Debug, Clone)]
pub struct MlpAgirliklari {
    /// `[2 * ffn, hidden]`: gate rows first, then value rows.
    pub wi: Vec<f32>,
    /// `[hidden, ffn]`.
    pub wo: Vec<f32>,
}

/// One layer, complete.
#[derive(Debug, Clone)]
pub struct KatmanAgirliklari {
    /// The kind of attention this layer runs.
    pub tur: KatmanTuru,
    /// Rope base for this layer.
    pub theta: f32,
    /// Attention branch.
    pub dikkat: DikkatAgirliklari,
    /// Feed-forward branch.
    pub mlp: MlpAgirliklari,
    /// Normalisation before the attention branch.
    ///
    /// `None` means the checkpoint has no tensor here and the branch runs on
    /// the unnormalised state. That is the case for layer 0 only.
    pub attn_norm: Option<Vec<f32>>,
    /// Normalisation before the feed-forward branch.
    pub mlp_norm: Vec<f32>,
}

/// Everything a forward pass needs, loaded once.
#[derive(Debug, Clone)]
pub struct Agirliklar {
    /// The configuration this checkpoint declared.
    pub yapi: KodlayiciYapisi,
    /// Where the embedding table lives, and how wide a row is.
    ///
    /// The table is *not* held in memory on the real path: it is 196 million
    /// elements, 786 MB once widened to `f32`, and this machine has less than
    /// that. Rows are read from the part files on demand - see
    /// [`Dizin::tensor_aralik_oku`] - which keeps a run's footprint at the size
    /// of the layer weights instead.
    pub gomme: GommeKaynagi,
    /// Normalisation applied to the embeddings.
    pub embeddings_norm: Vec<f32>,
    /// The layers, in order.
    pub katmanlar: Vec<KatmanAgirliklari>,
    /// Normalisation applied after the last layer.
    pub final_norm: Vec<f32>,
}

/// Where the embedding table is read from.
///
/// Two variants and not one: a real package is 786 MB of rows once widened to
/// `f32`, so a run reads them on demand; a test that exercises the *blocks*
/// must not need a 643 MB file, so it may hand over a table it built. The
/// variant is explicit rather than a flag, so the cheap path in a test cannot
/// be reached by accident in a run.
#[derive(Debug, Clone)]
pub enum GommeKaynagi {
    /// Rows come from the checkpoint's part files.
    Dosya {
        /// The reader.
        dosya: ParcaliDosya,
        /// The header, for offsets and element counts.
        dizin: Dizin,
    },
    /// Rows are already in memory. Tests only; a run on a real package uses
    /// [`GommeKaynagi::Dosya`].
    Bellek(Vec<f32>),
}

/// Where one layer's tensors live in the checkpoint.
fn katman_adlari(sira: usize) -> Vec<String> {
    let on = format!("encoder.layers.{sira}");
    vec![
        format!("{on}.attn.Wqkv.weight"),
        format!("{on}.attn.Wo.weight"),
        format!("{on}.mlp.Wi.weight"),
        format!("{on}.mlp.Wo.weight"),
        format!("{on}.attn_norm.weight"),
        format!("{on}.mlp_norm.weight"),
    ]
}

/// The names every load needs, so a missing tensor is reported before any work.
const ORTAK_ADLAR: [&str; 3] = [
    "encoder.embeddings.tok_embeddings.weight",
    "encoder.embeddings.norm.weight",
    "encoder.final_norm.weight",
];

/// The embedding tensor's name, used by the row reader.
const GOMLE_ADI: &str = "encoder.embeddings.tok_embeddings.weight";

/// Every tensor name this module reads, given a configuration.
///
/// The command subtracts this list from the header's to report tensors nothing
/// reads. A checkpoint whose weights are mostly unused is a checkpoint read
/// wrongly, and that is worth seeing before the first number is believed.
#[must_use]
pub fn kullanilan_adlar(yapi: &KodlayiciYapisi) -> Vec<String> {
    let mut adlar = vec![
        ORTAK_ADLAR[0].to_string(),
        ORTAK_ADLAR[1].to_string(),
        ORTAK_ADLAR[2].to_string(),
    ];
    for sira in 0..yapi.num_hidden_layers {
        let katman = katman_adlari(sira);
        for (i, ad) in katman.iter().enumerate() {
            // Layer 0's attention norm is absent by construction; not listing
            // it keeps the unused-tensor report honest rather than noisy.
            if i == 4 && sira == 0 {
                continue;
            }
            adlar.push(ad.clone());
        }
    }
    adlar
}

impl Agirliklar {
    /// Loads the whole encoder from a split checkpoint.
    ///
    /// Every name is resolved before the first byte is read, so a checkpoint
    /// that is missing a tensor fails in milliseconds rather than after
    /// hundreds of megabytes have been converted.
    ///
    /// # Errors
    /// [`BaslikHatasi`] for a missing name, a shape that is not what the
    /// configuration implies, or an unreadable range.
    pub fn yukle(
        yapi: KodlayiciYapisi,
        dosya: &ParcaliDosya,
        dizin: &Dizin,
    ) -> Result<Self, BaslikHatasi> {
        let mut eksik = Vec::new();
        for ad in ORTAK_ADLAR {
            if dizin.tensor(ad).is_none() {
                eksik.push(ad.to_string());
            }
        }
        // Layer 0's attention norm is deliberately absent; see the load rule
        // below. A pre-flight list that demanded it would refuse the real
        // checkpoint in the one place whose whole purpose is to name what is
        // missing.
        for sira in 0..yapi.num_hidden_layers {
            for (konum, ad) in katman_adlari(sira).into_iter().enumerate() {
                if konum == 4 && sira == 0 {
                    continue;
                }
                if dizin.tensor(&ad).is_none() {
                    eksik.push(ad);
                }
            }
        }
        if !eksik.is_empty() {
            // The first few names, not the whole list: a report a reader can
            // act on rather than a wall of names.
            let goster: Vec<&str> = eksik.iter().take(4).map(String::as_str).collect();
            return Err(BaslikHatasi::Baslik {
                mesaj: format!("{} tensor eksik: {}", eksik.len(), goster.join(", ")),
            });
        }

        let h = yapi.hidden_size;
        // The shape is checked from the header rather than by loading the
        // table: the check has to be cheap enough that it happens before the
        // expensive thing, not after it.
        let gomme_eleman = dizin
            .tensor(GOMLE_ADI)
            .map_or(0, crate::baslik::TensorBasligi::eleman);
        if gomme_eleman != yapi.vocab_size * h {
            return Err(BaslikHatasi::Baslik {
                mesaj: format!(
                    "tok_embeddings {} eleman, beklenen {}",
                    gomme_eleman,
                    yapi.vocab_size * h
                ),
            });
        }
        let embeddings_norm = dizin.tensor_oku(dosya, ORTAK_ADLAR[1])?;
        let final_norm = dizin.tensor_oku(dosya, ORTAK_ADLAR[2])?;

        let mut katmanlar = Vec::with_capacity(yapi.num_hidden_layers);
        for sira in 0..yapi.num_hidden_layers {
            let adlar = katman_adlari(sira);
            let wqkv = dizin.tensor_oku(dosya, &adlar[0])?;
            let wo = dizin.tensor_oku(dosya, &adlar[1])?;
            let wi = dizin.tensor_oku(dosya, &adlar[2])?;
            let mlp_wo = dizin.tensor_oku(dosya, &adlar[3])?;
            // The first layer has no attention norm, and that is by design:
            // the embedding block already ends with a norm, so the reference
            // architecture builds layer 0's attention norm as an identity and
            // the checkpoint therefore ships no tensor for it. Measured on the
            // real package: `encoder.layers.0.attn_norm.weight` is absent while
            // the other 21 layers all have it.
            //
            // The rule is enforced in both directions. A missing norm on a
            // later layer is an error, because running it as an identity would
            // produce plausible numbers from the wrong model. An *extra* norm on
            // layer 0 is ignored rather than refused, because the reference
            // ignores it too; it is reported as an unused tensor so nothing is
            // hidden.
            let attn_norm = if dizin.iceriyor(&adlar[4]) {
                Some(dizin.tensor_oku(dosya, &adlar[4])?)
            } else if sira == 0 {
                None
            } else {
                return Err(BaslikHatasi::Baslik {
                    mesaj: format!("katman {sira}: {} yok", adlar[4]),
                });
            };
            let mlp_norm = dizin.tensor_oku(dosya, &adlar[5])?;
            if wqkv.len() != 3 * h * h || wo.len() != h * h {
                return Err(BaslikHatasi::Baslik {
                    mesaj: format!("katman {sira}: dikkat sekli yapilandirmayla uyusmuyor"),
                });
            }
            if wi.len() % (2 * h) != 0 || mlp_wo.len() != h * (wi.len() / (2 * h)) {
                return Err(BaslikHatasi::Baslik {
                    mesaj: format!("katman {sira}: mlp sekli tutmuyor"),
                });
            }
            let tur = yapi.layer_types[sira];
            katmanlar.push(KatmanAgirliklari {
                tur,
                theta: yapi.theta(tur),
                dikkat: DikkatAgirliklari { wqkv, wo },
                mlp: MlpAgirliklari { wi, wo: mlp_wo },
                attn_norm,
                mlp_norm,
            });
        }
        Ok(Self {
            yapi,
            gomme: GommeKaynagi::Dosya {
                dosya: dosya.clone(),
                dizin: dizin.clone(),
            },
            embeddings_norm,
            katmanlar,
            final_norm,
        })
    }

    /// The embedding of one token id, read from the checkpoint.
    ///
    /// # Errors
    /// [`BaslikHatasi::AralikDisi`] for an id outside the vocabulary, which is
    /// refused rather than wrapped: a wrapped id returns a real vector for a
    /// token that does not exist.
    pub fn token_vektoru(&self, id: u32) -> Result<Vec<f32>, BaslikHatasi> {
        let h = self.yapi.hidden_size;
        let baslangic = id as usize * h;
        match &self.gomme {
            GommeKaynagi::Dosya { dosya, dizin } => {
                dizin.tensor_aralik_oku(dosya, GOMLE_ADI, baslangic, h)
            }
            GommeKaynagi::Bellek(tablo) => {
                if baslangic + h > tablo.len() {
                    return Err(BaslikHatasi::AralikDisi {
                        ad: format!("token {id}"),
                    });
                }
                Ok(tablo[baslangic..baslangic + h].to_vec())
            }
        }
    }
}

/// Whether a key position is inside a query's window.
///
/// # Why the window is a labelled rule and not a number
///
/// The checkpoint states the window *width* and not its *shape*, and the three
/// shapes that produce a "window of 128" are not the same function:
///
/// | rule | allowed keys for query `i` |
/// |---|---|
/// | [`PencereKurali::ReferansYarisi`] | everything whose distance to the left is under `w`; the future is not masked, which is what a lower-triangular mask with `diagonal = -w` produces |
/// | [`PencereKurali::SolPencere`] | the same, but the future is masked too |
/// | [`PencereKurali::Simetrik`] | `w` positions on either side |
///
/// The default is the first, because that is the shape the reference
/// implementation's published masking code produces, and the model card calls
/// the backbone bidirectional. It is a *labelled* parameter because this
/// environment cannot run the reference to settle it; a wrong window does not
/// crash, it returns slightly different numbers, and that is the kind of error
/// that has to be named rather than assumed away.
#[must_use]
pub fn pencere_icinde(hedef: usize, konum: usize, pencere: usize, kural: PencereKurali) -> bool {
    if pencere == 0 {
        return hedef == konum;
    }
    match kural {
        PencereKurali::ReferansYarisi => hedef + pencere > konum,
        PencereKurali::SolPencere => hedef <= konum && hedef + pencere > konum,
        PencereKurali::Simetrik => hedef.abs_diff(konum) < pencere,
    }
}

/// Which positions a sliding layer may see. See [`pencere_icinde`] for the
/// table of shapes and why this is a choice rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PencereKurali {
    /// Distance to the left under the width; the future is not masked.
    #[default]
    ReferansYarisi,
    /// The same, with the future masked as well.
    SolPencere,
    /// The width on either side.
    Simetrik,
}

/// Runs one layer over a sequence of hidden states.
///
/// `gizli` is `[uzunluk * hidden]`, row-major by position.
///
/// # Errors
/// [`BaslikHatasi::Baslik`] for a shape that does not fit the weights or the
/// configuration.
#[allow(clippy::too_many_lines)]
pub fn katman_ileri(
    gizli: &mut [f32],
    agirlik: &KatmanAgirliklari,
    yapi: &KodlayiciYapisi,
    uzunluk: usize,
) -> Result<(), BaslikHatasi> {
    let h = yapi.hidden_size;
    let kafa_sayisi = yapi.num_attention_heads;
    let kafa = yapi.kafa_genisligi();
    if gizli.len() != uzunluk * h || kafa == 0 {
        return Err(BaslikHatasi::Baslik {
            mesaj: format!("gizli {} eleman, beklenen {}", gizli.len(), uzunluk * h),
        });
    }
    if uzunluk == 0 {
        return Ok(());
    }

    // Branch 1: attention over the normalised input.
    let mut normal = gizli.to_vec();
    if let Some(norm) = &agirlik.attn_norm {
        let mut bas = 0;
        while bas < uzunluk {
            katman_norm(
                &mut normal[bas * h..(bas + 1) * h],
                norm,
                yapi.layer_norm_eps,
            )
            .map_err(|hata| BaslikHatasi::Baslik {
                mesaj: hata.to_string(),
            })?;
            bas += 1;
        }
    }

    // q, k, v for every position: three passes over one matrix, which is the
    // layout the checkpoint stores.
    let mut q = vec![0.0_f32; uzunluk * h];
    let mut k = vec![0.0_f32; uzunluk * h];
    let mut v = vec![0.0_f32; uzunluk * h];
    let mut gecici = vec![0.0_f32; 3 * h];
    for konum in 0..uzunluk {
        matris_vektor(
            &agirlik.dikkat.wqkv,
            &normal[konum * h..(konum + 1) * h],
            &mut gecici,
            h,
        )
        .map_err(|hata| BaslikHatasi::Baslik {
            mesaj: hata.to_string(),
        })?;
        q[konum * h..(konum + 1) * h].copy_from_slice(&gecici[..h]);
        k[konum * h..(konum + 1) * h].copy_from_slice(&gecici[h..2 * h]);
        v[konum * h..(konum + 1) * h].copy_from_slice(&gecici[2 * h..]);
    }

    // Rope, applied to the query and the key only: rotating the value would
    // change what is carried, not where it is attended from.
    let (kosin, sinus) = rope_tablosu(uzunluk, kafa, agirlik.theta);
    for konum in 0..uzunluk {
        for kafa_sirasi in 0..kafa_sayisi {
            let dilim = konum * h + kafa_sirasi * kafa;
            rope(&mut q[dilim..dilim + kafa], &kosin, &sinus, konum).map_err(|hata| {
                BaslikHatasi::Baslik {
                    mesaj: hata.to_string(),
                }
            })?;
            rope(&mut k[dilim..dilim + kafa], &kosin, &sinus, konum).map_err(|hata| {
                BaslikHatasi::Baslik {
                    mesaj: hata.to_string(),
                }
            })?;
        }
    }

    // Scores, mask, softmax, then the weighted sum of the values.
    //
    // The attention is **bidirectional**: this checkpoint is an encoder, and
    // the model card says so. A causal mask here would make every position
    // blind to what follows it, which for a masked-language model is not a
    // detail but a different function.
    let olcek = 1.0 / (kafa as f32).sqrt();
    let pencere = yapi.local_attention;
    let mut skorlar = vec![0.0_f32; uzunluk];
    let mut cikti = vec![0.0_f32; uzunluk * h];
    for konum in 0..uzunluk {
        for kafa_sirasi in 0..kafa_sayisi {
            let q_dilim = &q[konum * h + kafa_sirasi * kafa..konum * h + (kafa_sirasi + 1) * kafa];
            for (hedef, skor) in skorlar.iter_mut().enumerate() {
                if agirlik.tur == KatmanTuru::SlidingAttention
                    && !pencere_icinde(hedef, konum, pencere, yapi.pencere_kurali)
                {
                    // Outside the window: a mask, not a smaller loop, so that
                    // both layer kinds share one code path and the comparison
                    // between them means something.
                    *skor = f32::NEG_INFINITY;
                    continue;
                }
                let k_dilim =
                    &k[hedef * h + kafa_sirasi * kafa..hedef * h + (kafa_sirasi + 1) * kafa];
                let nokta = q_dilim
                    .iter()
                    .zip(k_dilim.iter())
                    .fold(0.0_f32, |acc, (a, b)| acc + a * b);
                *skor = nokta * olcek;
            }
            softmax(&mut skorlar).map_err(|hata| BaslikHatasi::Baslik {
                mesaj: hata.to_string(),
            })?;
            for (kaynak, agirlik_payi) in skorlar.iter().enumerate() {
                if *agirlik_payi == 0.0 {
                    continue;
                }
                let v_dilim =
                    &v[kaynak * h + kafa_sirasi * kafa..kaynak * h + (kafa_sirasi + 1) * kafa];
                for (i, deger) in v_dilim.iter().enumerate() {
                    cikti[konum * h + kafa_sirasi * kafa + i] += agirlik_payi * deger;
                }
            }
        }
    }

    // Projection and the first residual.
    let mut projeksiyon = vec![0.0_f32; h];
    for konum in 0..uzunluk {
        matris_vektor(
            &agirlik.dikkat.wo,
            &cikti[konum * h..(konum + 1) * h],
            &mut projeksiyon,
            h,
        )
        .map_err(|hata| BaslikHatasi::Baslik {
            mesaj: hata.to_string(),
        })?;
        for i in 0..h {
            gizli[konum * h + i] += projeksiyon[i];
        }
    }

    // Branch 2: the gated feed-forward, over a fresh normalisation of the
    // *residual-updated* state.
    let mut normal2 = gizli.to_vec();
    let ffn = agirlik.mlp.wi.len() / (2 * h);
    let mut kapi = vec![0.0_f32; ffn];
    let mut deger = vec![0.0_f32; ffn];
    let mut mlp_cikti = vec![0.0_f32; h];
    for konum in 0..uzunluk {
        katman_norm(
            &mut normal2[konum * h..(konum + 1) * h],
            &agirlik.mlp_norm,
            yapi.layer_norm_eps,
        )
        .map_err(|hata| BaslikHatasi::Baslik {
            mesaj: hata.to_string(),
        })?;
        let x = &normal2[konum * h..(konum + 1) * h];
        // The gate is the first `ffn` rows, the value the second `ffn` rows.
        for (sira, hedef) in kapi.iter_mut().enumerate() {
            let satir = &agirlik.mlp.wi[sira * h..(sira + 1) * h];
            *hedef = satir
                .iter()
                .zip(x.iter())
                .fold(0.0_f32, |a, (w, v)| a + w * v);
        }
        for (sira, hedef) in deger.iter_mut().enumerate() {
            let baslangic = (ffn + sira) * h;
            let satir = &agirlik.mlp.wi[baslangic..baslangic + h];
            *hedef = satir
                .iter()
                .zip(x.iter())
                .fold(0.0_f32, |a, (w, v)| a + w * v);
        }
        // GELU on the gate, multiply by the value: the product is what makes
        // this gated rather than a plain two-layer perceptron.
        let karisim: Vec<f32> = kapi
            .iter()
            .zip(deger.iter())
            .map(|(g, d)| gelu(*g) * *d)
            .collect();
        matris_vektor(&agirlik.mlp.wo, &karisim, &mut mlp_cikti, ffn).map_err(|hata| {
            BaslikHatasi::Baslik {
                mesaj: hata.to_string(),
            }
        })?;
        for i in 0..h {
            gizli[konum * h + i] += mlp_cikti[i];
        }
    }
    Ok(())
}

/// Runs the encoder over a token sequence and returns the final hidden states.
///
/// The output is the *pre-head* representation: `[uzunluk * hidden]`. Pooling
/// and the decision head are the caller's, because two callers that pool
/// differently must be able to share one forward pass.
///
/// # Errors
/// [`BaslikHatasi`] for an empty sequence, an unknown token id, or a shape that
/// does not fit.
pub fn kodla(agirliklar: &Agirliklar, kimlikler: &[u32]) -> Result<Vec<f32>, BaslikHatasi> {
    let h = agirliklar.yapi.hidden_size;
    if kimlikler.is_empty() {
        return Err(BaslikHatasi::Baslik {
            mesaj: "bos dizi: kodlanacak jeton yok".to_string(),
        });
    }
    if kimlikler.len() > agirliklar.yapi.max_position_embeddings.max(1) {
        return Err(BaslikHatasi::AralikDisi {
            ad: format!(
                "{} jeton, sinir {}",
                kimlikler.len(),
                agirliklar.yapi.max_position_embeddings
            ),
        });
    }
    let mut gizli = vec![0.0_f32; kimlikler.len() * h];
    for (sira, id) in kimlikler.iter().enumerate() {
        let vektor = agirliklar.token_vektoru(*id)?;
        gizli[sira * h..(sira + 1) * h].copy_from_slice(&vektor);
    }
    for konum in 0..kimlikler.len() {
        katman_norm(
            &mut gizli[konum * h..(konum + 1) * h],
            &agirliklar.embeddings_norm,
            agirliklar.yapi.layer_norm_eps,
        )
        .map_err(|hata| BaslikHatasi::Baslik {
            mesaj: hata.to_string(),
        })?;
    }
    for (sira, katman) in agirliklar.katmanlar.iter().enumerate() {
        katman_ileri(&mut gizli, katman, &agirliklar.yapi, kimlikler.len()).map_err(|hata| {
            BaslikHatasi::Baslik {
                mesaj: format!("katman {sira}: {hata:?}"),
            }
        })?;
    }
    for konum in 0..kimlikler.len() {
        katman_norm(
            &mut gizli[konum * h..(konum + 1) * h],
            &agirliklar.final_norm,
            agirliklar.yapi.layer_norm_eps,
        )
        .map_err(|hata| BaslikHatasi::Baslik {
            mesaj: hata.to_string(),
        })?;
    }
    Ok(gizli)
}

/// Mean pooling over positions, which is the pooling the classic head used.
///
/// Kept because it is a *choice*: mean pooling treats every position as
/// evidence, and a first-token pooling would make the representation depend on
/// a token whose meaning is a convention.
///
/// # Errors
/// [`BaslikHatasi::Baslik`] for a shape that does not divide by the hidden size.
pub fn ortalama_havuz(gizli: &[f32], hidden: usize) -> Result<Vec<f32>, BaslikHatasi> {
    if hidden == 0 || gizli.is_empty() || !gizli.len().is_multiple_of(hidden) {
        return Err(BaslikHatasi::Baslik {
            mesaj: format!("havuz: {} eleman, hidden {hidden}", gizli.len()),
        });
    }
    let konum = gizli.len() / hidden;
    let mut ortalama = vec![0.0_f32; hidden];
    for parca in gizli.chunks_exact(hidden) {
        for (i, v) in parca.iter().enumerate() {
            ortalama[i] += v;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n = konum as f32;
    for v in &mut ortalama {
        *v /= n;
    }
    Ok(ortalama)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yapilandirma::KatmanTuru as KT;

    fn kucuk_yapi(katman: usize) -> KodlayiciYapisi {
        KodlayiciYapisi {
            vocab_size: 8,
            hidden_size: 4,
            num_attention_heads: 2,
            intermediate_size: 6,
            num_hidden_layers: katman,
            layer_types: vec![KT::FullAttention; katman],
            rope_parameters: std::collections::BTreeMap::new(),
            local_attention: 4,
            pencere_kurali: PencereKurali::SolPencere,
            layer_norm_eps: 1.0e-5,
            max_position_embeddings: 16,
            hidden_activation: "gelu".to_string(),
            pad_token_id: Some(0),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            mask_token_id: Some(3),
            tie_word_embeddings: true,
            classifier_pooling: Some("mean".to_string()),
            diger: std::collections::BTreeMap::new(),
        }
    }

    fn kucuk_katman(yapi: &KodlayiciYapisi, tur: KT) -> KatmanAgirliklari {
        let h = yapi.hidden_size;
        let ffn = yapi.intermediate_size;
        // Deterministic weights from a fixed seed: a test that depends on the
        // system's randomness is a test that fails on another machine. The
        // generator matters as much as the seed - a linear ramp produces
        // nearly rank-one matrices, whose contribution is almost a scalar
        // multiple of the input, and after the final normalisation two
        // different stacks then look identical. That is measured, not assumed:
        // with a ramp the one-layer and three-layer outputs agreed to 5e-7.
        let doldur = |n: usize, kaydirma: u64| -> Vec<f32> {
            let mut tohum: u64 =
                0x2545_F491_4F6C_DD1D ^ kaydirma.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            (0..n)
                .map(|_| {
                    tohum ^= tohum << 13;
                    tohum ^= tohum >> 7;
                    tohum ^= tohum << 17;
                    #[allow(clippy::cast_precision_loss)]
                    let x = ((tohum >> 40) as f32 / 8_388_608.0 - 1.0) * 0.35;
                    x
                })
                .collect()
        };
        KatmanAgirliklari {
            tur,
            theta: 10_000.0,
            dikkat: DikkatAgirliklari {
                wqkv: doldur(3 * h * h, 1),
                wo: doldur(h * h, 2),
            },
            mlp: MlpAgirliklari {
                wi: doldur(2 * ffn * h, 3),
                wo: doldur(h * ffn, 4),
            },
            attn_norm: Some(vec![1.0; h]),
            mlp_norm: vec![1.0; h],
        }
    }

    /// A tiny in-memory checkpoint for the block tests.
    ///
    /// `Agirliklar` reads the embedding table from part files, so the tests
    /// that exercise the *blocks* need a way in that does not depend on a file.
    /// That is what `gomme_yolu` is: a table kept in memory, used by
    /// [`blok::token_vektoru_test`] and by nothing in production.
    fn kucuk_agirliklar(katman: usize) -> Agirliklar {
        let yapi = kucuk_yapi(katman);
        let h = yapi.hidden_size;
        let katmanlar = (0..katman)
            .map(|_| kucuk_katman(&yapi, KT::FullAttention))
            .collect();
        let mut tohum: u64 = 0x1234_5678_9ABC_DEF0;
        let gomme_tablosu: Vec<f32> = (0..yapi.vocab_size * h)
            .map(|_| {
                tohum ^= tohum << 13;
                tohum ^= tohum >> 7;
                tohum ^= tohum << 17;
                #[allow(clippy::cast_precision_loss)]
                let x = ((tohum >> 40) as f32 / 8_388_608.0 - 1.0) * 0.5;
                x
            })
            .collect();
        Agirliklar {
            // The test path keeps the embedding in memory: it is tiny here,
            // and the disk reader has its own tests in `baslik`.
            gomme: GommeKaynagi::Bellek(gomme_tablosu),
            embeddings_norm: vec![1.0; h],
            katmanlar,
            final_norm: vec![1.0; h],
            yapi,
        }
    }

    #[test]
    fn one_layer_produces_finite_numbers_of_the_right_length() {
        let agirliklar = kucuk_agirliklar(1);
        let cikti = kodla(&agirliklar, &[1, 2, 3]).expect("kodlanmali");
        assert_eq!(cikti.len(), 3 * 4);
        assert!(cikti.iter().all(|v| v.is_finite()), "{cikti:?}");
    }

    #[test]
    fn an_empty_sequence_is_refused_not_answered_with_a_zero_vector() {
        let agirliklar = kucuk_agirliklar(1);
        assert!(kodla(&agirliklar, &[]).is_err());
    }

    #[test]
    fn a_token_outside_the_vocabulary_is_refused_not_wrapped() {
        let agirliklar = kucuk_agirliklar(1);
        assert!(kodla(&agirliklar, &[7]).is_ok());
        assert!(kodla(&agirliklar, &[8]).is_err());
        assert!(kodla(&agirliklar, &[9999]).is_err());
    }

    #[test]
    fn attention_is_bidirectional_and_a_sliding_layer_is_bounded() {
        // The encoder is bidirectional: position 0's output depends on the
        // token at position 2. A causal mask would break exactly this.
        let agirliklar = kucuk_agirliklar(1);
        let a = kodla(&agirliklar, &[1, 2, 3]).expect("kodlanmali");
        let sonraki = kodla(&agirliklar, &[1, 2, 4]).expect("kodlanmali");
        let ilk_degisti = a[..4]
            .iter()
            .zip(sonraki[..4].iter())
            .any(|(x, y)| (x - y).abs() > 1e-9);
        assert!(
            ilk_degisti,
            "tam (cift yonlu) dikkatte ilk konum sonraki konumdan etkilenmeli"
        );

        // With a symmetric window of 1, position 0 sees only itself, so a
        // change at position 2 cannot reach it.
        let mut yapi = kucuk_yapi(1);
        yapi.local_attention = 1;
        yapi.pencere_kurali = PencereKurali::Simetrik;
        let mut agirliklar = kucuk_agirliklar(1);
        agirliklar.yapi = yapi;
        agirliklar.katmanlar[0].tur = KT::SlidingAttention;
        let b = kodla(&agirliklar, &[1, 2, 3]).expect("kodlanmali");
        let c = kodla(&agirliklar, &[1, 2, 4]).expect("kodlanmali");
        assert!(
            (b[0] - c[0]).abs() < 1e-6,
            "pencere 1'de ilk konum sonraki konumdan etkilenmemeli: {} vs {}",
            b[0],
            c[0]
        );
    }

    #[test]
    fn the_three_window_shapes_agree_on_the_recent_left_and_differ_elsewhere() {
        // All three rules allow the query itself and the recent past.
        for kural in [
            PencereKurali::ReferansYarisi,
            PencereKurali::SolPencere,
            PencereKurali::Simetrik,
        ] {
            assert!(
                pencere_icinde(10, 10, 4, kural),
                "{kural:?} kendini gormeli"
            );
            assert!(
                pencere_icinde(7, 10, 4, kural),
                "{kural:?} yakin gecmisi gormeli"
            );
            assert!(
                !pencere_icinde(6, 10, 4, kural),
                "{kural:?} uzak gecmisi gormemeli"
            );
        }
        // The future: only the reference shape leaves it open.
        assert!(pencere_icinde(11, 10, 4, PencereKurali::ReferansYarisi));
        assert!(!pencere_icinde(11, 10, 4, PencereKurali::SolPencere));
        assert!(pencere_icinde(13, 10, 4, PencereKurali::Simetrik));
        assert!(!pencere_icinde(14, 10, 4, PencereKurali::Simetrik));
        // A width of one still sees the query under every rule, and a width of
        // zero is the query alone rather than nothing at all.
        for kural in [
            PencereKurali::ReferansYarisi,
            PencereKurali::SolPencere,
            PencereKurali::Simetrik,
        ] {
            assert!(pencere_icinde(10, 10, 1, kural));
            assert!(pencere_icinde(10, 10, 0, kural));
            assert!(!pencere_icinde(9, 10, 0, kural));
        }
    }

    #[test]
    fn the_gate_and_the_value_halves_are_not_interchangeable() {
        // Swapping the halves of Wi must change the output; if it did not, the
        // split point would be untested and a port could read them backwards.
        let mut agirliklar = kucuk_agirliklar(1);
        let a = kodla(&agirliklar, &[1, 2, 3]).expect("kodlanmali");
        let h = agirliklar.yapi.hidden_size;
        let ffn = agirliklar.yapi.intermediate_size;
        let wi = agirliklar.katmanlar[0].mlp.wi.clone();
        let mut yeni = wi.clone();
        for satir in 0..ffn {
            for i in 0..h {
                yeni[satir * h + i] = wi[(ffn + satir) * h + i];
                yeni[(ffn + satir) * h + i] = wi[satir * h + i];
            }
        }
        agirliklar.katmanlar[0].mlp.wi = yeni;
        let b = kodla(&agirliklar, &[1, 2, 3]).expect("kodlanmali");
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    #[test]
    fn a_deeper_stack_changes_the_output_and_stays_finite() {
        let tek = kodla(&kucuk_agirliklar(1), &[1, 2, 3]).expect("kodlanmali");
        let uc = kodla(&kucuk_agirliklar(3), &[1, 2, 3]).expect("kodlanmali");
        assert!(uc.iter().all(|v| v.is_finite()));
        assert!(tek.iter().zip(uc.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    #[test]
    fn the_same_input_twice_gives_the_same_answer() {
        let agirliklar = kucuk_agirliklar(2);
        let a = kodla(&agirliklar, &[1, 2, 3, 4]).expect("kodlanmali");
        let b = kodla(&agirliklar, &[1, 2, 3, 4]).expect("kodlanmali");
        assert_eq!(a, b);
    }

    #[test]
    fn mean_pooling_averages_and_refuses_a_ragged_shape() {
        let gizli = vec![1.0_f32, 2.0, 3.0, 4.0];
        let o = ortalama_havuz(&gizli, 2).expect("havuzlanmali");
        assert_eq!(o, vec![2.0, 3.0]);
        assert!(ortalama_havuz(&gizli, 3).is_err());
        assert!(ortalama_havuz(&[], 2).is_err());
    }

    #[test]
    fn loading_reports_missing_tensors_before_reading_any_bytes() {
        let yapi = kucuk_yapi(1);
        // A directory with one tiny part: the load must fail on the *name*
        // check, which happens before any tensor is read.
        let klasor = std::env::temp_dir().join("lubot-kodlayici-eksik");
        let _ = std::fs::remove_dir_all(&klasor);
        std::fs::create_dir_all(&klasor).expect("klasor");
        let baslik = r#"{"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut govde = Vec::new();
        govde.extend_from_slice(&(baslik.len() as u64).to_le_bytes());
        govde.extend_from_slice(baslik.as_bytes());
        govde.extend_from_slice(&[0_u8; 2]);
        std::fs::write(klasor.join("model.safetensors.part-00"), &govde).expect("yazilmali");
        let dosya = ParcaliDosya::ac(&klasor, "model.safetensors.part-").expect("acilmali");
        let dizin = Dizin::oku(&dosya).expect("baslik");
        let hata = Agirliklar::yukle(yapi, &dosya, &dizin).unwrap_err();
        match hata {
            BaslikHatasi::Baslik { mesaj } => assert!(mesaj.contains("eksik"), "{mesaj}"),
            digeri => panic!("eksik tensor bekleniyordu: {digeri:?}"),
        }
        let _ = std::fs::remove_dir_all(&klasor);
    }
}
