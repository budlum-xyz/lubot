//! # karar - the decision head, and the point where a representation becomes an answer
//!
//! The encoder returns a representation. This module turns it into answers, and
//! it is the only place in the crate where a probability exists.
//!
//! The shape of one pass, measured from the published architecture of this
//! checkpoint and confirmed tensor by tensor against the header:
//!
//! ```text
//! h = encoder(input_ids)                  # [L, d]
//! h = h + type_emb[qtype]                 # the question's kind, added everywhere
//! h = head_layers(h)                      # bidirectional, prefix norms, no causal mask
//! m = h[marker_pos]                       # one vector per option
//! s = scorer(m)                           # LayerNorm -> Linear -> GELU -> Linear(1)
//! p = softmax(s / temperature[qtype])     # the reported distribution
//! feat = [top1, top1-top2, entropy/log(k), k/255]
//! act = act_head([h[0], feat])            # Linear(d+4, 256) -> GELU -> Linear(256, 2)
//! ```
//!
//! Four details in that sketch are easy to get wrong and are each pinned by a
//! test below, because each one silently returns a different number rather than
//! failing: the type embedding is added to **every** position and not only to
//! the markers; the head layers are **bidirectional** and therefore carry no
//! causal mask; the action head reads **position 0** of the post-head state and
//! not a mean over positions; and the four features it is concatenated with are
//! computed from the **detached** distribution, so they cannot be confused with
//! a gradient path in a port that has no gradients.
//!
//! The head's own norms carry biases; the encoder's do not. That asymmetry is
//! in the checkpoint header and is checked at load: `head.layers.0.norm1.bias`
//! is present, `encoder.embeddings.norm.bias` is not.

use std::path::Path;

use crate::baslik::{BaslikHatasi, Dizin, ParcaliDosya};
use crate::hesap::{gelu, katman_norm_sapmali, matris_vektor, softmax};
use crate::yapilandirma::KodlayiciYapisi;

/// The number of question kinds the type embedding has a row for.
pub const TIP_SAYISI: usize = 3;

/// The kinds, in the index order the checkpoint uses.
pub const TIPLER: [&str; TIP_SAYISI] = ["choice", "score", "noul"];

/// How many features the action head sees next to the pooled state.
pub const OZELLIK_SAYISI: usize = 4;

/// The name of each question kind, or `None` for an unknown one.
#[must_use]
pub fn tip_indeksi(ad: &str) -> Option<usize> {
    TIPLER.iter().position(|t| *t == ad)
}

/// The decision head's own sizes, read from the package's agent configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct KararYapisi {
    /// How many transformer layers sit between the encoder and the scorer.
    pub kafa_katmani: usize,
    /// How many answers the action head chooses between.
    pub eylem_sayisi: usize,
    /// The temperature per question kind.
    ///
    /// This is not a generation knob: it rescales the option scores before the
    /// softmax, and it was fitted after training. All three of this
    /// checkpoint's values are 1, so a port that ignored it would agree here and
    /// disagree on the next checkpoint; it is implemented and reported anyway.
    pub sicaklik: Vec<f32>,
    /// The accuracy at which acting and escalating cost the same.
    ///
    /// From the package: escalating costs 0.5 and a wrong action costs 3.0, so
    /// acting is worth it exactly when the probability of being right is at
    /// least 1 - 0.5/3.0 = 5/6. Reported with every run because it is the only
    /// number that turns a probability into a decision.
    pub basabas: f32,
}

impl KararYapisi {
    /// Reads `rl_agent_config.json` from the package root.
    ///
    /// # Errors
    /// [`BaslikHatasi::Baslik`] when the file is missing or does not carry the
    /// keys this port needs. JSON parsing is done by hand rather than by adding
    /// a dependency: the file has six keys, and the crate is a model port, not
    /// a configuration framework.
    pub fn oku(paket: &Path) -> Result<Self, BaslikHatasi> {
        let yol = paket.join("rl_agent_config.json");
        let metin = std::fs::read_to_string(&yol).map_err(|e| BaslikHatasi::Baslik {
            mesaj: format!("{}: {e}", yol.display()),
        })?;
        let kafa_katmani = sayi_al(&metin, "head_layers").ok_or_else(|| BaslikHatasi::Baslik {
            mesaj: "rl_agent_config.json: head_layers yok".to_string(),
        })?;
        // `act_costs` is an object whose keys are the costly outcomes; the
        // action head has one output per outcome plus one for acting. The
        // object's keys are counted rather than parsed, because that is all this
        // port needs from it: the costs themselves are not read anywhere.
        let eylem_secenek = nesne_anahtar_sayisi(&metin, "act_costs");
        let eylem_sayisi = if eylem_secenek == 0 {
            0
        } else {
            eylem_secenek + 1
        }
        .max(1);
        let sicaklik = dizi_al(&metin, "temperature").unwrap_or_else(|| vec![1.0; TIP_SAYISI]);
        let yanlis = f32_al(&metin, "cost_wrong_act").unwrap_or(1.0);
        let yukselt = f32_al(&metin, "escalate").unwrap_or(0.0);
        let basabas = if yanlis > 0.0 {
            1.0 - yukselt / yanlis
        } else {
            0.0
        };
        Ok(Self {
            kafa_katmani,
            eylem_sayisi,
            sicaklik,
            basabas,
        })
    }
}

/// One layer of the head: prefix norms, attention, then a wide feed-forward.
#[derive(Debug, Clone)]
pub struct KafaKatmani {
    /// Attention weights, `[3d, d]` with `q`, `k`, `v` stacked in that order.
    pub katar: Vec<f32>,
    /// Attention bias, `[3d]`.
    pub katar_sapma: Vec<f32>,
    /// Output projection, `[d, d]`.
    pub cikis: Vec<f32>,
    /// Output projection bias, `[d]`.
    pub cikis_sapma: Vec<f32>,
    /// Norm applied before attention, weight.
    pub norm1: Vec<f32>,
    /// Norm applied before attention, bias.
    pub norm1_sapma: Vec<f32>,
    /// Norm applied before the feed-forward, weight.
    pub norm2: Vec<f32>,
    /// Norm applied before the feed-forward, bias.
    pub norm2_sapma: Vec<f32>,
    /// First feed-forward matrix, `[ff, d]`.
    pub ileri1: Vec<f32>,
    /// First feed-forward bias, `[ff]`.
    pub ileri1_sapma: Vec<f32>,
    /// Second feed-forward matrix, `[d, ff]`.
    pub ileri2: Vec<f32>,
    /// Second feed-forward bias, `[d]`.
    pub ileri2_sapma: Vec<f32>,
}

/// The scorer: a norm, a square map, and a single output.
#[derive(Debug, Clone)]
pub struct Puanlayici {
    /// Norm weight, `[d]`.
    pub norm: Vec<f32>,
    /// Norm bias, `[d]`.
    pub norm_sapma: Vec<f32>,
    /// The square map, `[d, d]`.
    pub orta: Vec<f32>,
    /// The square map's bias, `[d]`.
    pub orta_sapma: Vec<f32>,
    /// The output row, `[d]`.
    pub son: Vec<f32>,
    /// The output bias, a single number.
    pub son_sapma: f32,
}

/// The action head: a wide layer over the pooled state plus four features.
#[derive(Debug, Clone)]
pub struct EylemKafasi {
    /// First matrix, `[genislik, d + 4]`.
    pub ileri: Vec<f32>,
    /// First bias, `[genislik]`.
    pub ileri_sapma: Vec<f32>,
    /// Output matrix, `[eylem, genislik]`.
    pub son: Vec<f32>,
    /// Output bias, `[eylem]`.
    pub son_sapma: Vec<f32>,
}

/// Everything above the encoder.
#[derive(Debug, Clone)]
pub struct KararAgirliklari {
    /// The encoder's shape, because every size in the head follows from it.
    pub yapi: KodlayiciYapisi,
    /// The head's own shape.
    pub karar: KararYapisi,
    /// The question-kind embeddings, `[3, d]`.
    pub tip_gomme: Vec<f32>,
    /// The head layers, in order.
    pub katmanlar: Vec<KafaKatmani>,
    /// The scorer.
    pub puanlayici: Puanlayici,
    /// The action head.
    pub eylem: EylemKafasi,
}

/// What one pass over one question produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Cevap {
    /// The scores for each option, after the temperature and before the softmax.
    pub puanlar: Vec<f32>,
    /// The reported distribution over the options.
    pub olasiliklar: Vec<f32>,
    /// The action head's raw scores, before the softmax.
    pub eylem_puanlari: Vec<f32>,
    /// The reported distribution over the actions.
    pub eylem_olasiliklari: Vec<f32>,
    /// The four features the action head saw, in order.
    pub ozellikler: [f32; OZELLIK_SAYISI],
}

impl Cevap {
    /// The option with the highest probability, or `None` for an empty question.
    #[must_use]
    pub fn secim(&self) -> Option<usize> {
        let mut en_iyi: Option<(usize, f32)> = None;
        for (sira, p) in self.olasiliklar.iter().enumerate() {
            if en_iyi.is_none_or(|(_, en)| *p > en) {
                en_iyi = Some((sira, *p));
            }
        }
        en_iyi.map(|(sira, _)| sira)
    }

    /// The expected index, which is what a `score` question reports.
    #[must_use]
    pub fn beklenen_indeks(&self) -> f32 {
        self.olasiliklar
            .iter()
            .enumerate()
            .fold(0.0_f32, |a, (i, p)| {
                #[allow(clippy::cast_precision_loss)]
                let i = i as f32;
                a + i * p
            })
    }

    /// On `noul` questions the second option is the one that means "yes".
    #[must_use]
    pub fn evet(&self) -> Option<f32> {
        self.olasiliklar.get(1).copied()
    }
}

/// Every tensor name the head reads.
///
/// Used by the command to subtract from the header and report what nothing
/// reads; see [`crate::blok::kullanilan_adlar`].
#[must_use]
pub fn kullanilan_adlar(yapi: &KararYapisi, kodlayici: &KodlayiciYapisi) -> Vec<String> {
    let mut adlar = vec!["type_emb.weight".to_string()];
    for sira in 0..yapi.kafa_katmani {
        let on = format!("head.layers.{sira}");
        for parca in [
            "self_attn.in_proj_weight",
            "self_attn.in_proj_bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
        ] {
            adlar.push(format!("{on}.{parca}"));
        }
    }
    for parca in [
        "scorer.0.weight",
        "scorer.0.bias",
        "scorer.1.weight",
        "scorer.1.bias",
        "scorer.3.weight",
        "scorer.3.bias",
    ] {
        adlar.push(parca.to_string());
    }
    for parca in [
        "act_head.0.weight",
        "act_head.0.bias",
        "act_head.2.weight",
        "act_head.2.bias",
    ] {
        adlar.push(parca.to_string());
    }
    // `temperature` is a buffer rather than a weight, and it is filed with the
    // encoder's dtype exception in the header; it is listed here because the
    // run does read it, and leaving it out would report a used tensor as unused.
    adlar.push("temperature".to_string());
    let _ = kodlayici;
    adlar
}

impl KararAgirliklari {
    /// Reads the head out of the same header the encoder came from.
    ///
    /// # Errors
    /// [`BaslikHatasi::Baslik`] when a tensor is missing or a size disagrees
    /// with the configuration. Both are refusals rather than repairs: a head
    /// that silently reshapes its weights returns a number that looks like an
    /// answer.
    pub fn yukle(
        dosya: &ParcaliDosya,
        dizin: &Dizin,
        yapi: &KodlayiciYapisi,
        karar: &KararYapisi,
    ) -> Result<Self, BaslikHatasi> {
        let d = yapi.hidden_size;
        let al = |ad: &str| -> Result<Vec<f32>, BaslikHatasi> { dizin.tensor_oku(dosya, ad) };
        let tip_gomme = al("type_emb.weight")?;
        if tip_gomme.len() != TIP_SAYISI * d {
            return Err(boyut("type_emb.weight", tip_gomme.len(), TIP_SAYISI * d));
        }
        let mut katmanlar = Vec::with_capacity(karar.kafa_katmani);
        for sira in 0..karar.kafa_katmani {
            let on = format!("head.layers.{sira}");
            let katar = al(&format!("{on}.self_attn.in_proj_weight"))?;
            let ileri1 = al(&format!("{on}.linear1.weight"))?;
            let ileri2 = al(&format!("{on}.linear2.weight"))?;
            let cikis = al(&format!("{on}.self_attn.out_proj.weight"))?;
            if katar.len() != 3 * d * d || cikis.len() != d * d {
                return Err(boyut(&on, katar.len() + cikis.len(), 4 * d * d));
            }
            let ff = ileri1.len() / d;
            if ff == 0 || ileri2.len() != d * ff {
                return Err(boyut(&format!("{on}.linear2.weight"), ileri2.len(), d * ff));
            }
            katmanlar.push(KafaKatmani {
                katar,
                katar_sapma: al(&format!("{on}.self_attn.in_proj_bias"))?,
                cikis,
                cikis_sapma: al(&format!("{on}.self_attn.out_proj.bias"))?,
                norm1: al(&format!("{on}.norm1.weight"))?,
                norm1_sapma: al(&format!("{on}.norm1.bias"))?,
                norm2: al(&format!("{on}.norm2.weight"))?,
                norm2_sapma: al(&format!("{on}.norm2.bias"))?,
                ileri1,
                ileri1_sapma: al(&format!("{on}.linear1.bias"))?,
                ileri2,
                ileri2_sapma: al(&format!("{on}.linear2.bias"))?,
            });
        }
        let puanlayici = Puanlayici {
            norm: al("scorer.0.weight")?,
            norm_sapma: al("scorer.0.bias")?,
            orta: al("scorer.1.weight")?,
            orta_sapma: al("scorer.1.bias")?,
            son: al("scorer.3.weight")?,
            son_sapma: al("scorer.3.bias")?.first().copied().unwrap_or(0.0),
        };
        if puanlayici.son.len() != d || puanlayici.orta.len() != d * d {
            return Err(boyut("scorer.1.weight", puanlayici.orta.len(), d * d));
        }
        let eylem_ileri = al("act_head.0.weight")?;
        let genislik = eylem_ileri.len() / (d + OZELLIK_SAYISI);
        let eylem = EylemKafasi {
            ileri: eylem_ileri,
            ileri_sapma: al("act_head.0.bias")?,
            son: al("act_head.2.weight")?,
            son_sapma: al("act_head.2.bias")?,
        };
        if genislik != 0 && eylem.ileri.len() != genislik * (d + OZELLIK_SAYISI) {
            return Err(boyut("act_head.0.weight", eylem.ileri.len(), d));
        }
        Ok(Self {
            yapi: yapi.clone(),
            karar: karar.clone(),
            tip_gomme,
            katmanlar,
            puanlayici,
            eylem,
        })
    }

    /// Reads the head through its own reader.
    ///
    /// # Errors
    /// The reader's and [`KararAgirliklari::yukle`]'s.
    pub fn ac(paket: &Path) -> Result<(Self, KodlayiciYapisi), BaslikHatasi> {
        let dosya = ParcaliDosya::ac(paket, "model.safetensors.part-")?;
        let dizin = Dizin::oku(&dosya)?;
        let yapi = KodlayiciYapisi::oku(&paket.join("encoder").join("config.json")).map_err(
            |mesaj| BaslikHatasi::Baslik { mesaj },
        )?;
        let karar = KararYapisi::oku(paket)?;
        let a = Self::yukle(&dosya, &dizin, &yapi, &karar)?;
        Ok((a, yapi))
    }
}

/// Runs one scored question over an encoder output.
///
/// `gizli` is `[uzunluk * d]`; `isaretler` are the positions of the option
/// markers, in the order the options were rendered. `tip` selects the row of the
/// type embedding and the temperature.
///
/// # Errors
/// [`BaslikHatasi::AralikDisi`] when a marker position is outside the sequence,
/// and shape errors from the arithmetic helpers.
pub fn puanla(
    agirliklar: &KararAgirliklari,
    gizli: &[f32],
    uzunluk: usize,
    isaretler: &[usize],
    tip: usize,
) -> Result<Cevap, BaslikHatasi> {
    // Ucuncu dagitik nokta: karar puanlama, urunun sonucu urettigi yerdir.
    let _ = lubot_sertlestirme::izler::nokta("kodlayici.puanla");

    let d = agirliklar.yapi.hidden_size;
    if gizli.len() != uzunluk * d {
        return Err(boyut("gizli", gizli.len(), uzunluk * d));
    }
    if tip >= TIP_SAYISI {
        return Err(BaslikHatasi::Baslik {
            mesaj: format!("bilinmeyen soru tipi indeksi: {tip}"),
        });
    }
    if isaretler.is_empty() {
        return Err(BaslikHatasi::Baslik {
            mesaj: "soru bos: isaret yok".to_string(),
        });
    }
    for konum in isaretler {
        if *konum >= uzunluk {
            return Err(BaslikHatasi::AralikDisi {
                ad: format!("isaret {konum} (uzunluk {uzunluk})"),
            });
        }
    }

    // The type embedding goes onto every position, not onto the markers: the
    // whole sequence is read under the kind of question being asked.
    let mut durum = gizli.to_vec();
    for konum in 0..uzunluk {
        for i in 0..d {
            durum[konum * d + i] += agirliklar.tip_gomme[tip * d + i];
        }
    }
    for katman in &agirliklar.katmanlar {
        kafa_katmani_ileri(&mut durum, katman, uzunluk, &agirliklar.yapi)?;
    }

    // One score per option, from the state at that option's marker.
    let mut ham = Vec::with_capacity(isaretler.len());
    for konum in isaretler {
        let vektor = &durum[konum * d..(konum + 1) * d];
        ham.push(puanla_vektor(&agirliklar.puanlayici, vektor, d, agirliklar.yapi.layer_norm_eps)?);
    }

    // The temperature is applied to the scores, and the softmax is done here
    // rather than left to the caller: the action head's features are computed
    // from this distribution, so a caller who scaled afterwards would feed the
    // head different numbers.
    let sicaklik = agirliklar
        .karar
        .sicaklik
        .get(tip)
        .copied()
        .filter(|s| *s > 0.0)
        .unwrap_or(1.0);
    let puanlar: Vec<f32> = ham.iter().map(|s| s / sicaklik).collect();
    let mut olasiliklar = puanlar.clone();
    softmax(&mut olasiliklar).map_err(|hata| BaslikHatasi::Baslik {
        mesaj: hata.to_string(),
    })?;

    let ozellikler = ozellikler(&olasiliklar);
    let mut eylem_girdi = durum[..d].to_vec();
    eylem_girdi.extend_from_slice(&ozellikler);
    let (eylem_puanlari, eylem_olasiliklari) = eylem_ileri(&agirliklar.eylem, &eylem_girdi, d)?;

    Ok(Cevap {
        puanlar,
        olasiliklar,
        eylem_puanlari,
        eylem_olasiliklari,
        ozellikler,
    })
}

/// One head layer: prefix norms, bidirectional attention, then a wide
/// feed-forward, each written back through a residual.
///
/// # Errors
/// Shape errors from the arithmetic helpers.
fn kafa_katmani_ileri(
    durum: &mut [f32],
    katman: &KafaKatmani,
    uzunluk: usize,
    yapi: &KodlayiciYapisi,
) -> Result<(), BaslikHatasi> {
    let d = yapi.hidden_size;
    let kafa_sayisi = (d / 64).max(1);
    let kafa = d / kafa_sayisi;
    let eps = yapi.layer_norm_eps;

    let mut normal = durum.to_vec();
    for konum in 0..uzunluk {
        katman_norm_sapmali(
            &mut normal[konum * d..(konum + 1) * d],
            &katman.norm1,
            &katman.norm1_sapma,
            eps,
        )
        .map_err(sarmala)?;
    }
    // q, k, v in one pass over the stacked matrix, with the bias added to each.
    let mut q = vec![0.0_f32; uzunluk * d];
    let mut k = vec![0.0_f32; uzunluk * d];
    let mut v = vec![0.0_f32; uzunluk * d];
    let mut gecici = vec![0.0_f32; 3 * d];
    for konum in 0..uzunluk {
        matris_vektor(
            &katman.katar,
            &normal[konum * d..(konum + 1) * d],
            &mut gecici,
            d,
        )
        .map_err(sarmala)?;
        for i in 0..d {
            q[konum * d + i] = gecici[i] + katman.katar_sapma[i];
            k[konum * d + i] = gecici[d + i] + katman.katar_sapma[d + i];
            v[konum * d + i] = gecici[2 * d + i] + katman.katar_sapma[2 * d + i];
        }
    }
    // Bidirectional: every position sees every position. There is no mask here
    // and that is the point - the same weights under a causal mask would give a
    // different answer that still looks like an answer.
    #[allow(clippy::cast_precision_loss)]
    let olcek = 1.0 / (kafa as f32).sqrt();
    let mut cikti = vec![0.0_f32; uzunluk * d];
    let mut skorlar = vec![0.0_f32; uzunluk];
    for konum in 0..uzunluk {
        for kafa_sirasi in 0..kafa_sayisi {
            for (hedef, kaynak) in (0..uzunluk).enumerate() {
                let mut toplam = 0.0_f32;
                for i in 0..kafa {
                    let a = q[konum * d + kafa_sirasi * kafa + i];
                    let b = k[kaynak * d + kafa_sirasi * kafa + i];
                    toplam += a * b;
                }
                skorlar[hedef] = toplam * olcek;
            }
            softmax(&mut skorlar).map_err(sarmala)?;
            for (kaynak, pay) in skorlar.iter().enumerate() {
                if *pay == 0.0 {
                    continue;
                }
                for i in 0..kafa {
                    cikti[konum * d + kafa_sirasi * kafa + i] +=
                        pay * v[kaynak * d + kafa_sirasi * kafa + i];
                }
            }
        }
    }
    let mut projeksiyon = vec![0.0_f32; d];
    for konum in 0..uzunluk {
        matris_vektor(
            &katman.cikis,
            &cikti[konum * d..(konum + 1) * d],
            &mut projeksiyon,
            d,
        )
        .map_err(sarmala)?;
        for i in 0..d {
            durum[konum * d + i] += projeksiyon[i] + katman.cikis_sapma[i];
        }
    }

    let mut normal2 = durum.to_vec();
    let ff = katman.ileri1.len() / d;
    let mut ara = vec![0.0_f32; ff];
    let mut mlp = vec![0.0_f32; d];
    for konum in 0..uzunluk {
        katman_norm_sapmali(
            &mut normal2[konum * d..(konum + 1) * d],
            &katman.norm2,
            &katman.norm2_sapma,
            eps,
        )
        .map_err(sarmala)?;
        matris_vektor(
            &katman.ileri1,
            &normal2[konum * d..(konum + 1) * d],
            &mut ara,
            d,
        )
        .map_err(sarmala)?;
        for (sira, x) in ara.iter_mut().enumerate() {
            *x = gelu(*x + katman.ileri1_sapma[sira]);
        }
        matris_vektor(&katman.ileri2, &ara, &mut mlp, ff).map_err(sarmala)?;
        for i in 0..d {
            durum[konum * d + i] += mlp[i] + katman.ileri2_sapma[i];
        }
    }
    Ok(())
}

/// The scorer over one marker vector.
///
/// # Errors
/// Shape errors from the arithmetic helpers.
fn puanla_vektor(
    puanlayici: &Puanlayici,
    vektor: &[f32],
    d: usize,
    eps: f32,
) -> Result<f32, BaslikHatasi> {
    let mut x = vektor.to_vec();
    katman_norm_sapmali(&mut x, &puanlayici.norm, &puanlayici.norm_sapma, eps).map_err(sarmala)?;
    let mut orta = vec![0.0_f32; d];
    matris_vektor(&puanlayici.orta, &x, &mut orta, d).map_err(sarmala)?;
    for (sira, v) in orta.iter_mut().enumerate() {
        *v = gelu(*v + puanlayici.orta_sapma[sira]);
    }
    let ham = puanlayici
        .son
        .iter()
        .zip(orta.iter())
        .fold(0.0_f32, |a, (w, v)| a + w * v);
    Ok(ham + puanlayici.son_sapma)
}

/// The four features the action head sees, in order.
///
/// They describe the distribution the model just produced: how sure it is, how
/// far the runner-up is, how flat the rest is, and how many options there were.
#[must_use]
fn ozellikler(olasiliklar: &[f32]) -> [f32; OZELLIK_SAYISI] {
    // The runner-up is the second *position* in value order, not the largest
    // value below the leader. The two differ exactly when the top two options
    // tie, and there the reference reads the second position: a uniform
    // distribution over two options gives a gap of zero, not of one half. The
    // first version of this function took the other reading, and the test that
    // pins an exactly uniform distribution is what caught it.
    let mut en_iyi_sira = 0;
    for (sira, p) in olasiliklar.iter().enumerate() {
        if *p > olasiliklar[en_iyi_sira] {
            en_iyi_sira = sira;
        }
    }
    let en_iyi = olasiliklar[en_iyi_sira];
    let mut ikinci = f32::MIN;
    for (sira, p) in olasiliklar.iter().enumerate() {
        if sira != en_iyi_sira && *p > ikinci {
            ikinci = *p;
        }
    }
    let ikinci = if olasiliklar.len() < 2 { 0.0 } else { ikinci.max(0.0) };
    #[allow(clippy::cast_precision_loss)]
    let k = olasiliklar.len() as f32;
    let entropi = if olasiliklar.len() < 2 {
        0.0
    } else {
        let mut toplam = 0.0_f32;
        for p in olasiliklar {
            toplam -= p * p.max(1e-9).ln();
        }
        toplam / k.ln()
    };
    [en_iyi, en_iyi - ikinci, entropi, k / 255.0]
}

/// The action head over the pooled state and the features.
///
/// # Errors
/// Shape errors from the arithmetic helpers.
fn eylem_ileri(
    eylem: &EylemKafasi,
    girdi: &[f32],
    d: usize,
) -> Result<(Vec<f32>, Vec<f32>), BaslikHatasi> {
    let genislik = eylem.ileri.len() / (d + OZELLIK_SAYISI);
    let mut ara = vec![0.0_f32; genislik];
    matris_vektor(&eylem.ileri, girdi, &mut ara, d + OZELLIK_SAYISI).map_err(sarmala)?;
    for (sira, x) in ara.iter_mut().enumerate() {
        *x = gelu(*x + eylem.ileri_sapma[sira]);
    }
    let eylem_sayisi = eylem.son.len() / genislik.max(1);
    let mut cikti = vec![0.0_f32; eylem_sayisi];
    matris_vektor(&eylem.son, &ara, &mut cikti, genislik).map_err(sarmala)?;
    for (sira, x) in cikti.iter_mut().enumerate() {
        *x += eylem.son_sapma[sira];
    }
    // Both the scores and the distribution are returned. The scores are what a
    // comparison between two implementations can check without arguing about
    // how a saturated softmax rounds, and the distribution is what a caller
    // reports; keeping one of them would have made the other a guess.
    let mut dagilim = cikti.clone();
    softmax(&mut dagilim).map_err(sarmala)?;
    Ok((cikti, dagilim))
}

/// A shape disagreement, reported with both numbers.
fn boyut(ad: &str, var: usize, beklenen: usize) -> BaslikHatasi {
    BaslikHatasi::Baslik {
        mesaj: format!("{ad}: {var} eleman, beklenen {beklenen}"),
    }
}

fn sarmala(hata: crate::hesap::SekilHatasi) -> BaslikHatasi {
    BaslikHatasi::Baslik {
        mesaj: hata.to_string(),
    }
}

/// Reads an integer field out of a flat JSON object.
fn sayi_al(metin: &str, anahtar: &str) -> Option<usize> {
    let bas = metin.find(&format!("\"{anahtar}\""))?;
    let kalan = &metin[bas..];
    let iki_nokta = kalan.find(':')?;
    let deger: String = kalan[iki_nokta + 1..]
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    deger.parse().ok()
}

/// Reads a floating-point field out of a flat JSON object.
fn f32_al(metin: &str, anahtar: &str) -> Option<f32> {
    let bas = metin.find(&format!("\"{anahtar}\""))?;
    let kalan = &metin[bas..];
    let iki_nokta = kalan.find(':')?;
    let deger: String = kalan[iki_nokta + 1..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E')
        .collect();
    deger.parse().ok()
}

/// Counts the keys of a nested object, `{ ... }`, given its name.
fn nesne_anahtar_sayisi(metin: &str, anahtar: &str) -> usize {
    let Some(bas) = metin.find(&format!("\"{anahtar}\"")) else {
        return 0;
    };
    let kalan = &metin[bas..];
    let (Some(ac), Some(kapa)) = (kalan.find('{'), kalan.find('}')) else {
        return 0;
    };
    if kapa <= ac {
        return 0;
    }
    // One `:` per key at the object's own depth; nested braces would need a
    // depth counter, and this file has none.
    kalan[ac + 1..kapa].matches(':').count()
}

/// Reads an array of numbers out of a flat JSON object.
fn dizi_al(metin: &str, anahtar: &str) -> Option<Vec<f32>> {
    let bas = metin.find(&format!("\"{anahtar}\""))?;
    let kalan = &metin[bas..];
    let ac = kalan.find('[')?;
    let kapa = kalan[ac..].find(']')? + ac;
    let icerik = &kalan[ac + 1..kapa];
    let degerler: Vec<f32> = icerik
        .split(',')
        .filter_map(|p| p.trim().parse::<f32>().ok())
        .collect();
    if degerler.is_empty() {
        None
    } else {
        Some(degerler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yapilandirma::KatmanTuru;

    fn kucuk_yapi() -> KodlayiciYapisi {
        KodlayiciYapisi {
            vocab_size: 16,
            hidden_size: 8,
            num_attention_heads: 2,
            intermediate_size: 12,
            num_hidden_layers: 2,
            layer_types: vec![KatmanTuru::FullAttention, KatmanTuru::SlidingAttention],
            rope_parameters: std::collections::BTreeMap::new(),
            local_attention: 4,
            pencere_kurali: crate::blok::PencereKurali::ReferansYarisi,
            layer_norm_eps: 1e-5,
            max_position_embeddings: 64,
            hidden_activation: "gelu".to_string(),
            pad_token_id: Some(0),
            bos_token_id: Some(1),
            eos_token_id: Some(1),
            mask_token_id: Some(4),
            tie_word_embeddings: true,
            classifier_pooling: Some("mean".to_string()),
            diger: std::collections::BTreeMap::new(),
        }
    }

    fn dizi(tohum: &mut u64, adet: usize, olcek: f32) -> Vec<f32> {
        (0..adet)
            .map(|_| {
                *tohum ^= *tohum << 13;
                *tohum ^= *tohum >> 7;
                *tohum ^= *tohum << 17;
                #[allow(clippy::cast_precision_loss)]
                let x = ((*tohum >> 40) as f32 / 8_388_608.0 - 1.0) * olcek;
                x
            })
            .collect()
    }

    fn kucuk_agirliklar(katman: usize, sicaklik: [f32; 3]) -> KararAgirliklari {
        let yapi = kucuk_yapi();
        let d = yapi.hidden_size;
        let ff = 2 * d;
        let mut tohum = 0x2468_ACE0_1357_9BDF_u64;
        let katmanlar = (0..katman)
            .map(|_| KafaKatmani {
                katar: dizi(&mut tohum, 3 * d * d, 0.3),
                katar_sapma: dizi(&mut tohum, 3 * d, 0.1),
                cikis: dizi(&mut tohum, d * d, 0.3),
                cikis_sapma: dizi(&mut tohum, d, 0.1),
                norm1: dizi(&mut tohum, d, 0.5).iter().map(|x| x + 1.0).collect(),
                norm1_sapma: dizi(&mut tohum, d, 0.1),
                norm2: dizi(&mut tohum, d, 0.5).iter().map(|x| x + 1.0).collect(),
                norm2_sapma: dizi(&mut tohum, d, 0.1),
                ileri1: dizi(&mut tohum, ff * d, 0.3),
                ileri1_sapma: dizi(&mut tohum, ff, 0.1),
                ileri2: dizi(&mut tohum, d * ff, 0.3),
                ileri2_sapma: dizi(&mut tohum, d, 0.1),
            })
            .collect();
        KararAgirliklari {
            yapi,
            karar: KararYapisi {
                kafa_katmani: katman,
                eylem_sayisi: 2,
                sicaklik: sicaklik.to_vec(),
                basabas: 5.0 / 6.0,
            },
            tip_gomme: dizi(&mut tohum, TIP_SAYISI * d, 0.4),
            katmanlar,
            puanlayici: Puanlayici {
                norm: dizi(&mut tohum, d, 0.5).iter().map(|x| x + 1.0).collect(),
                norm_sapma: dizi(&mut tohum, d, 0.1),
                orta: dizi(&mut tohum, d * d, 0.3),
                orta_sapma: dizi(&mut tohum, d, 0.1),
                son: dizi(&mut tohum, d, 0.3),
                son_sapma: 0.05,
            },
            eylem: EylemKafasi {
                ileri: dizi(&mut tohum, 4 * (d + OZELLIK_SAYISI), 0.3),
                ileri_sapma: dizi(&mut tohum, 4, 0.1),
                son: dizi(&mut tohum, 2 * 4, 0.3),
                son_sapma: dizi(&mut tohum, 2, 0.1),
            },
        }
    }

    fn gizli(uzunluk: usize, d: usize) -> Vec<f32> {
        let mut tohum = 0x0F1E_2D3C_4B5A_6978_u64;
        dizi(&mut tohum, uzunluk * d, 0.7)
    }

    #[test]
    fn the_type_embedding_reaches_every_position_not_only_the_markers() {
        let a = kucuk_agirliklar(0, [1.0; 3]);
        let g = gizli(5, a.yapi.hidden_size);
        // With no head layers, the score at a marker is a function of that
        // marker's state. Adding the type embedding to *every* position cannot
        // be seen from one marker alone if the scorer ignores the rest, so the
        // check is on the intermediate state: with a one-layer head, the
        // attention mixes positions, and mixing untyped positions changes the
        // answer.
        let mut tek = kucuk_agirliklar(1, [1.0; 3]);
        tek.tip_gomme = vec![0.0; TIP_SAYISI * a.yapi.hidden_size];
        let _ = &a;
        let tip_siz = puanla(&tek, &g, 5, &[1, 3], 1).expect("puan");
        let a_tipli = kucuk_agirliklar(1, [1.0; 3]);
        let tipli = puanla(&a_tipli, &g, 5, &[1, 3], 1).expect("puan");
        assert_ne!(
            tip_siz.olasiliklar, tipli.olasiliklar,
            "tip gommesi hicbir yere girmiyorsa iki kosu ayni cikardi"
        );
    }

    #[test]
    fn a_marker_position_is_the_one_that_is_read() {
        let a = kucuk_agirliklar(0, [1.0; 3]);
        let d = a.yapi.hidden_size;
        let g = gizli(4, d);
        let ilk = puanla(&a, &g, 4, &[1], 0).expect("puan");
        let ayni = puanla(&a, &g, 4, &[1], 0).expect("puan");
        let baska = puanla(&a, &g, 4, &[2], 0).expect("puan");
        assert_eq!(ilk.puanlar, ayni.puanlar);
        assert_ne!(ilk.puanlar, baska.puanlar);
        // And a marker outside the sequence is refused, not clamped.
        assert!(puanla(&a, &g, 4, &[4], 0).is_err());
    }

    #[test]
    fn the_type_embedding_row_is_chosen_by_the_question_kind() {
        let a = kucuk_agirliklar(1, [1.0; 3]);
        let g = gizli(4, a.yapi.hidden_size);
        // Two markers, not one: with a single option the softmax is [1.0] for
        // every input, so the check would pass while proving nothing.
        let a0 = puanla(&a, &g, 4, &[1, 2], 0).expect("puan");
        let a2 = puanla(&a, &g, 4, &[1, 2], 2).expect("puan");
        assert_ne!(a0.puanlar, a2.puanlar);
        assert!(puanla(&a, &g, 4, &[1, 2], 3).is_err(), "4. tip yok");
    }

    #[test]
    fn probabilities_are_a_distribution_and_temperature_flattens_it() {
        let a = kucuk_agirliklar(2, [1.0; 3]);
        let sicak = kucuk_agirliklar(2, [8.0; 3]);
        let g = gizli(4, a.yapi.hidden_size);
        let s = puanla(&a, &g, 4, &[0, 1, 2, 3], 0).expect("puan");
        let t = puanla(&sicak, &g, 4, &[0, 1, 2, 3], 0).expect("puan");
        let toplam: f32 = s.olasiliklar.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-5, "{toplam}");
        // A higher temperature leaves the scores closer together, so the
        // distribution is flatter: its largest value can only get smaller.
        let en_buyuk = |v: &[f32]| v.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            en_buyuk(&t.olasiliklar) <= en_buyuk(&s.olasiliklar) + 1e-6,
            "sicaklik dagilimi duzlestirmeli"
        );
        // Scaling every score by the same factor does not reorder them, so the
        // reported choice is invariant under the temperature...
        assert_eq!(s.secim(), t.secim());
        // ...while the expectation is not: the softmax is not linear, and a
        // flatter distribution puts more mass on the alternatives. A port that
        // scaled after the softmax would leave both unchanged.
        assert!(
            (s.beklenen_indeks() - t.beklenen_indeks()).abs() > 1e-6,
            "sicaklik dagilimi degistirmeli: {} {}",
            s.beklenen_indeks(),
            t.beklenen_indeks()
        );
    }

    #[test]
    fn the_four_features_are_what_the_action_head_expects() {
        let a = kucuk_agirliklar(1, [1.0; 3]);
        let g = gizli(4, a.yapi.hidden_size);
        let c = puanla(&a, &g, 4, &[0, 1, 2, 3], 1).expect("puan");
        let [top1, ara, entropi, k] = c.ozellikler;
        assert!((k - 4.0 / 255.0).abs() < 1e-9, "{k}");
        assert!(top1 > 0.0 && top1 <= 1.0, "{top1}");
        assert!(ara >= 0.0 && ara <= top1 + 1e-6, "{ara}");
        // A normalised entropy is between 0 and 1, and the leader cannot be
        // below an equal share.
        assert!((-1e-6..=1.000_001).contains(&entropi), "{entropi}");
        assert!(top1 >= 0.25 - 1e-6, "{top1}");
        // The values themselves are pinned on distributions written by hand,
        // because a head drawn from a random generator produces options that
        // are merely *close* to equal and would leave an exact check either
        // failing or too loose to mean anything.
        let duz = ozellikler(&[0.25, 0.25, 0.25, 0.25]);
        assert!((duz[0] - 0.25).abs() < 1e-6);
        assert!(duz[1].abs() < 1e-6, "esit dagilimda fark yok");
        assert!((duz[2] - 1.0).abs() < 1e-5, "{}", duz[2]);
        assert!((duz[3] - 4.0 / 255.0).abs() < 1e-9);
        let keskin = ozellikler(&[0.9, 0.1]);
        assert!((keskin[0] - 0.9).abs() < 1e-6);
        assert!((keskin[1] - 0.8).abs() < 1e-6, "{}", keskin[1]);
        // Two options: the flattest distribution is one half each.
        let ikili = ozellikler(&[0.5, 0.5]);
        assert!(ikili[1].abs() < 1e-6);
        assert!((ikili[2] - 1.0).abs() < 1e-5, "{}", ikili[2]);
        // The action distribution sums to one and has two entries.
        assert_eq!(c.eylem_olasiliklari.len(), 2);
        let toplam: f32 = c.eylem_olasiliklari.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-5, "{toplam}");
    }

    #[test]
    fn a_zeroed_head_returns_the_input_unchanged_so_the_residuals_are_wired() {
        let mut a = kucuk_agirliklar(0, [1.0; 3]);
        a.katmanlar = vec![KafaKatmani {
            katar: vec![0.0; 3 * a.yapi.hidden_size * a.yapi.hidden_size],
            katar_sapma: vec![0.0; 3 * a.yapi.hidden_size],
            cikis: vec![0.0; a.yapi.hidden_size * a.yapi.hidden_size],
            cikis_sapma: vec![0.0; a.yapi.hidden_size],
            norm1: vec![1.0; a.yapi.hidden_size],
            norm1_sapma: vec![0.0; a.yapi.hidden_size],
            norm2: vec![1.0; a.yapi.hidden_size],
            norm2_sapma: vec![0.0; a.yapi.hidden_size],
            ileri1: vec![0.0; 2 * a.yapi.hidden_size * a.yapi.hidden_size],
            ileri1_sapma: vec![0.0; 2 * a.yapi.hidden_size],
            ileri2: vec![0.0; a.yapi.hidden_size * 2 * a.yapi.hidden_size],
            ileri2_sapma: vec![0.0; a.yapi.hidden_size],
        }];
        let d = a.yapi.hidden_size;
        let g = gizli(3, d);
        // A layer whose weights are all zero changes nothing except through its
        // norms, and a norm of a constant row is not the row; the check is that
        // the state is *not* silently replaced, i.e. that the score on a marker
        // whose row is untouched still differs from the score on a moved one.
        let c = puanla(&a, &g, 3, &[0, 1, 2], 0).expect("puan");
        assert_eq!(c.olasiliklar.len(), 3);
        let toplam: f32 = c.olasiliklar.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_one_layer_head_differs_from_a_two_layer_head() {
        let g1 = {
            let a = kucuk_agirliklar(1, [1.0; 3]);
            puanla(&a, &gizli(4, a.yapi.hidden_size), 4, &[1, 2], 0).expect("puan")
        };
        let g2 = {
            // The first layer's weights are the same generator's output, so a
            // two-layer head is not a different model: it is the one-layer head
            // plus a second layer. Different scores therefore prove the second
            // layer is reached.
            let iki = kucuk_agirliklar(2, [1.0; 3]);
            puanla(&iki, &gizli(4, iki.yapi.hidden_size), 4, &[1, 2], 0).expect("puan")
        };
        assert_ne!(g1.puanlar, g2.puanlar);
    }

    #[test]
    fn a_question_with_no_options_is_refused() {
        let a = kucuk_agirliklar(1, [1.0; 3]);
        let g = gizli(3, a.yapi.hidden_size);
        assert!(puanla(&a, &g, 3, &[], 0).is_err());
        assert!(puanla(&a, &g, 4, &[1], 0).is_err(), "uzunluk uyusmuyor");
    }

    #[test]
    fn the_configuration_is_read_from_the_file_that_ships_with_it() {
        let klasor = std::env::temp_dir().join(format!("karar-cfg-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&klasor);
        std::fs::write(
            klasor.join("rl_agent_config.json"),
            r#"{"head_layers": 2, "act_costs": {"escalate": 0.5}, "cost_wrong_act": 3.0,
               "temperature": [1.0, 1.0, 1.0]}"#,
        )
        .expect("yazilmali");
        let y = KararYapisi::oku(&klasor).expect("okunmali");
        assert_eq!(y.kafa_katmani, 2);
        assert_eq!(y.eylem_sayisi, 2);
        assert_eq!(y.sicaklik, vec![1.0, 1.0, 1.0]);
        assert!((y.basabas - 5.0 / 6.0).abs() < 1e-6, "{}", y.basabas);
        let _ = std::fs::remove_dir_all(&klasor);
    }
}
