//! The training core: a forward pass, a loss, a backward pass, and the epoch
//! discipline around them. All of it written here - no framework, no upstream
//! weights, no autograd library (K1).
//!
//! # Why the gradient check is the point
//!
//! A hand-written backward pass is the easiest place in a from-scratch trainer
//! to be quietly wrong: the loss goes down anyway, the model learns something,
//! and the thing it learns is shaped by a wrong gradient. So correctness here
//! is not argued, it is measured - [`gradients_match_finite_differences`]
//! compares every analytic gradient against a central finite difference on a
//! small configuration and refuses the trainer when the relative error is above
//! [`GRADIENT_CHECK_TOLERANCE`].
//!
//! # The architecture is the spec's, not a guess
//!
//! [`Spec::lubot_a1`] reproduces `training/model_spec.json` exactly: 8192 × 64
//! tied embedding with a `1/d_model` logit scale, 8 pre-norm layers of 2-head
//! attention and a 64→256 MLP, `1/d_k` attention scale. [`Spec::parametre_sayisi`]
//! counts the parameters the spec claims (924.288) and a test holds the two
//! together, so the trainer cannot drift from the spec it is supposed to train.
//!
//! # Epochs are the grant crate's business
//!
//! The ceiling on how many epochs may run is not restated here. [`egitim_turu`]
//! asks [`lubot_grant`] for it, because one protocol constant written in two
//! places is a disagreement waiting to happen.

use lubot_grant::training::MAX_TRAINING_GRANT_EPOCHS;

pub mod dikkat_gruplu;
pub mod dongu;
pub mod kernel32;
pub mod kontrol;
pub mod kosu;
pub mod olcum;
pub mod veri;

/// Relative error above which the backward pass is considered wrong.
pub const GRADIENT_CHECK_TOLERANCE: f64 = 1e-6;
/// Absolute floor under the relative check.
///
/// A central finite difference cannot resolve a gradient whose magnitude is
/// near the noise floor: with a step of 1e-5 and a loss of order 1, the
/// round-off in `(k1 - k2) / 2h` is around 1e-10 absolute, which is a *large*
/// relative error on a gradient of 1e-6 and no signal at all about the
/// derivative. Measured here: on the tiny configuration the analytic and the
/// numeric value of the worst-scoring parameter agree to four significant
/// digits (-1.280e-6 against -1.280e-6) while their naive relative error reads
/// 8.6e-5. So the check is relative where the gradient is resolvable and
/// absolute where it is not, instead of reporting noise as a wrong gradient.
pub const GRADIENT_CHECK_MUTLAK_TABAN: f64 = 1e-9;
/// LayerNorm epsilon.
pub const LN_EPS: f64 = 1e-5;

/// The architecture, as the spec states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spec {
    /// Vocabulary size.
    pub vocab: usize,
    /// Model width.
    pub d_model: usize,
    /// Transformer layers.
    pub n_layers: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// Key/value heads. When it equals `n_heads` every query head owns its key
    /// and value; when it divides `n_heads`, a group of `n_heads / n_kv_heads`
    /// query heads shares one key/value pair. The default is full attention,
    /// so a spec that does not name the field's value has not changed.
    pub n_kv_heads: usize,
    /// MLP inner width.
    pub d_ff: usize,
    /// Longest sequence the model is built for, as the spec states it.
    pub max_seq_len: usize,
}

/// Why a spec was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecHatasi {
    /// A zero dimension.
    BosBoyut,
    /// The width does not divide by the head count.
    BasSayisiBolmuyor,
    /// The head count does not divide by the key/value head count.
    KvBasSayisiBolunmuyor,
}

impl Spec {
    /// The first training spec (`lubot-a1-derin-dar`).
    #[must_use]
    pub const fn lubot_a1() -> Self {
        Self {
            vocab: 8192,
            d_model: 64,
            n_layers: 8,
            n_heads: 2,
            n_kv_heads: 2,
            d_ff: 256,
            max_seq_len: 256,
        }
    }

    /// Width per head.
    #[must_use]
    pub fn d_k(self) -> usize {
        self.d_model / self.n_heads
    }

    /// Total key/value width: `n_kv_heads` heads of `d_k` each.
    ///
    /// Equal to `d_model` when every query head owns its key and value, which
    /// is why the full-attention spec's shapes and counts do not move.
    #[must_use]
    pub fn d_kv(self) -> usize {
        self.n_kv_heads * self.d_k()
    }

    /// # Errors
    /// [`SpecHatasi::BosBoyut`] on a zero dimension;
    /// [`SpecHatasi::BasSayisiBolmuyor`] when heads do not divide the width;
    /// [`SpecHatasi::KvBasSayisiBolunmuyor`] when the key/value head count
    /// does not divide the head count.
    pub fn dogrula(self) -> Result<(), SpecHatasi> {
        if self.vocab == 0 || self.d_model == 0 || self.n_layers == 0 || self.d_ff == 0 {
            return Err(SpecHatasi::BosBoyut);
        }
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return Err(SpecHatasi::BasSayisiBolmuyor);
        }
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(SpecHatasi::KvBasSayisiBolunmuyor);
        }
        Ok(())
    }

    /// How many parameters this spec has, counted the way the spec counts them:
    /// tied embedding once, attention and MLP per layer, two LayerNorms per
    /// layer and one at the end. Key and value projections count at the
    /// key/value width [`Spec::d_kv`], which is `d_model` itself when every
    /// head owns its key and value - so the full-attention count is the same
    /// number it has always been.
    #[must_use]
    pub fn parametre_sayisi(self) -> usize {
        let d = self.d_model;
        let kv = self.d_kv();
        let embedding = self.vocab * d;
        let dikkat = self.n_layers * (2 * d * d + 2 * d * kv + 2 * d + 2 * kv);
        let mlp = self.n_layers * (2 * d * self.d_ff + self.d_ff + d);
        let ln = self.n_layers * 4 * d + 2 * d;
        embedding + dikkat + mlp + ln
    }
}

/// Every weight, in one place.
#[derive(Debug, Clone, PartialEq)]
pub struct Parametreler {
    /// Tied token embedding / readout, `[vocab * d_model]`.
    pub embedding: Vec<f64>,
    /// Per layer: LayerNorm 1 scale, bias.
    pub ln1_olcek: Vec<f64>,
    /// Per layer: LayerNorm 1 bias.
    pub ln1_sapma: Vec<f64>,
    /// Per layer: query weights, `[n_layers * d_model * d_model]`.
    pub wq: Vec<f64>,
    /// Per layer: query biases.
    pub bq: Vec<f64>,
    /// Per layer: key weights, `[n_layers * d_model * d_kv]`.
    pub wk: Vec<f64>,
    /// Per layer: key biases, `[n_layers * d_kv]`.
    pub bk: Vec<f64>,
    /// Per layer: value weights, `[n_layers * d_model * d_kv]`.
    pub wv: Vec<f64>,
    /// Per layer: value biases, `[n_layers * d_kv]`.
    pub bv: Vec<f64>,
    /// Per layer: output projection weights.
    pub wo: Vec<f64>,
    /// Per layer: output projection biases.
    pub bo: Vec<f64>,
    /// Per layer: LayerNorm 2 scale.
    pub ln2_olcek: Vec<f64>,
    /// Per layer: LayerNorm 2 bias.
    pub ln2_sapma: Vec<f64>,
    /// Per layer: MLP up weights, `[n_layers * d_ff * d_model]`.
    pub w1: Vec<f64>,
    /// Per layer: MLP up biases.
    pub b1: Vec<f64>,
    /// Per layer: MLP down weights, `[n_layers * d_model * d_ff]`.
    pub w2: Vec<f64>,
    /// Per layer: MLP down biases.
    pub b2: Vec<f64>,
    /// Final LayerNorm scale.
    pub lnf_olcek: Vec<f64>,
    /// Final LayerNorm bias.
    pub lnf_sapma: Vec<f64>,
}

impl Parametreler {
    /// Zeroed gradients of the same shape.
    #[must_use]
    pub fn sifir_gradyan(&self) -> Parametreler {
        Self {
            embedding: vec![0.0; self.embedding.len()],
            ln1_olcek: vec![0.0; self.ln1_olcek.len()],
            ln1_sapma: vec![0.0; self.ln1_sapma.len()],
            wq: vec![0.0; self.wq.len()],
            bq: vec![0.0; self.bq.len()],
            wk: vec![0.0; self.wk.len()],
            bk: vec![0.0; self.bk.len()],
            wv: vec![0.0; self.wv.len()],
            bv: vec![0.0; self.bv.len()],
            wo: vec![0.0; self.wo.len()],
            bo: vec![0.0; self.bo.len()],
            ln2_olcek: vec![0.0; self.ln2_olcek.len()],
            ln2_sapma: vec![0.0; self.ln2_sapma.len()],
            w1: vec![0.0; self.w1.len()],
            b1: vec![0.0; self.b1.len()],
            w2: vec![0.0; self.w2.len()],
            b2: vec![0.0; self.b2.len()],
            lnf_olcek: vec![0.0; self.lnf_olcek.len()],
            lnf_sapma: vec![0.0; self.lnf_sapma.len()],
        }
    }

    /// Deterministic pseudo-random fill in `[-1, 1]`, so a test does not depend
    /// on a lucky draw. Not an initialiser: the μP init scale is the spec's and
    /// belongs to the run that uses it.
    #[must_use]
    pub fn belirgin_doldur(spec: Spec, tohum: u64) -> Self {
        let mut durum = tohum
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let mut sonraki = move || {
            durum = durum
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((durum >> 33) as f64) / ((1u64 << 31) as f64) * 2.0 - 1.0
        };
        let d = spec.d_model;
        let kv = spec.d_kv();
        let katman = spec.n_layers;
        Self {
            embedding: (0..spec.vocab * d).map(|_| sonraki() * 0.1).collect(),
            ln1_olcek: vec![1.0; katman * d],
            ln1_sapma: vec![0.0; katman * d],
            wq: (0..katman * d * d).map(|_| sonraki() * 0.1).collect(),
            bq: vec![0.0; katman * d],
            wk: (0..katman * d * kv).map(|_| sonraki() * 0.1).collect(),
            bk: vec![0.0; katman * kv],
            wv: (0..katman * d * kv).map(|_| sonraki() * 0.1).collect(),
            bv: vec![0.0; katman * kv],
            wo: (0..katman * d * d).map(|_| sonraki() * 0.1).collect(),
            bo: vec![0.0; katman * d],
            ln2_olcek: vec![1.0; katman * d],
            ln2_sapma: vec![0.0; katman * d],
            w1: (0..katman * spec.d_ff * d)
                .map(|_| sonraki() * 0.1)
                .collect(),
            b1: vec![0.0; katman * spec.d_ff],
            w2: (0..katman * d * spec.d_ff)
                .map(|_| sonraki() * 0.1)
                .collect(),
            b2: vec![0.0; katman * d],
            lnf_olcek: vec![1.0; d],
            lnf_sapma: vec![0.0; d],
        }
    }
}

/// One sequence in, one loss and its gradients out.
#[must_use]
pub fn ileri_ve_geri(
    spec: Spec,
    p: &Parametreler,
    girdi: &[usize],
    hedef: &[usize],
) -> (f64, Parametreler) {
    // Tek kayit: her konum ayni parcadan, yani sinir maskesi hicbir seyi
    // elemiyor. Paketli pencere icin `ileri_ve_geri_paket` kullanilir.
    let kaynak = vec![0u32; girdi.len()];
    ileri_ve_geri_paket(spec, p, girdi, hedef, &kaynak)
}

/// Forward and backward over a packed window, where each position carries the
/// record it came from.
///
/// Attention does not cross a record boundary, in the forward pass or in the
/// backward pass. That is what makes packing usable at all: without it a window
/// that joins two records would let a passage be predicted from a source it
/// cannot be attributed to.
///
/// # Panics
/// If `kaynak` is not the same length as `girdi`, or either is empty.
pub fn ileri_ve_geri_paket(
    spec: Spec,
    p: &Parametreler,
    girdi: &[usize],
    hedef: &[usize],
    kaynak: &[u32],
) -> (f64, Parametreler) {
    assert_eq!(
        kaynak.len(),
        girdi.len(),
        "kaynak vektoru girdiyle ayni uzunlukta olmali"
    );
    let d = spec.d_model;
    let t = girdi.len();
    let mut grad = p.sifir_gradyan();

    // Embedding lookup: x[t] = embedding[token[t]].
    let mut x = vec![0.0f64; t * d];
    for (i, tok) in girdi.iter().enumerate() {
        x[i * d..(i + 1) * d].copy_from_slice(&p.embedding[tok * d..(tok + 1) * d]);
    }

    let mut caches: Vec<KatmanBellek> = Vec::with_capacity(spec.n_layers);
    for l in 0..spec.n_layers {
        let (y, bellek) = katman_ileri(spec, p, l, &x, kaynak);
        x = y;
        caches.push(bellek);
    }

    // Final LayerNorm.
    let (mut xn, son_ortalama, son_rstd) = layer_norm_ileri(&x, d, t, &p.lnf_olcek, &p.lnf_sapma);

    // Tied readout with the spec's 1/d_model logit scale, then softmax + CE.
    let olcek = 1.0 / (d as f64);
    let mut toplam_kayip = 0.0;
    let mut dxn = vec![0.0f64; t * d];
    for i in 0..t {
        let mut logits = vec![0.0f64; spec.vocab];
        let xn_satir = &xn[i * d..(i + 1) * d];
        for (v, logit) in logits.iter_mut().enumerate() {
            let satir = &p.embedding[v * d..(v + 1) * d];
            *logit = satir.iter().zip(xn_satir).map(|(a, b)| a * b).sum::<f64>() * olcek;
        }
        let (kayip, mut dlogits) = softmax_ce(&logits, hedef[i]);
        toplam_kayip += kayip;
        // d/dembedding from the readout, and d/dxn.
        let dxn_satir = &mut dxn[i * d..(i + 1) * d];
        for (v, dlogit) in dlogits.iter().enumerate() {
            // The loss is averaged over positions, so is this contribution:
            // without the 1/t the tied readout would outweigh the input
            // embedding by a factor of t.
            let g = dlogit * olcek / (t as f64);
            if g == 0.0 {
                continue;
            }
            let grad_satir = &mut grad.embedding[v * d..(v + 1) * d];
            let emb_satir = &p.embedding[v * d..(v + 1) * d];
            for ((grad_deger, dxn_deger), (emb_deger, xn_deger)) in grad_satir
                .iter_mut()
                .zip(dxn_satir.iter_mut())
                .zip(emb_satir.iter().zip(xn_satir))
            {
                *grad_deger += g * *xn_deger;
                *dxn_deger += g * *emb_deger;
            }
        }
        dlogits.clear();
    }
    // `g` above already carries the 1/t of the averaged loss, so `dxn` is on
    // the right scale here; dividing again would make every upstream gradient
    // a factor of t too small.
    let kayip = toplam_kayip / (t as f64);

    // Final LayerNorm backward.
    let (dx, dg, db) = layer_norm_geri(&dxn, &xn, &x, d, t, &son_ortalama, &son_rstd, &p.lnf_olcek);
    for (i, g) in dg.iter().enumerate() {
        grad.lnf_olcek[i] += g;
    }
    for (i, g) in db.iter().enumerate() {
        grad.lnf_sapma[i] += g;
    }
    xn.clear();
    let mut dx_akis = dx;

    for l in (0..spec.n_layers).rev() {
        dx_akis = katman_geri(spec, p, &mut grad, l, &caches[l], &dx_akis, kaynak);
    }

    // Input embedding gradient: the tied matrix also feeds the readout.
    for (i, tok) in girdi.iter().enumerate() {
        for j in 0..d {
            grad.embedding[tok * d + j] += dx_akis[i * d + j];
        }
    }
    (kayip, grad)
}

/// What one layer needs to run backward.
struct KatmanBellek {
    girdi: Vec<f64>,
    ln1: Vec<f64>,
    ortalama1: Vec<f64>,
    rstd1: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    agirlik: Vec<f64>,
    attn: Vec<f64>,
    kalinti1: Vec<f64>,
    ln2: Vec<f64>,
    ortalama2: Vec<f64>,
    rstd2: Vec<f64>,
    on: Vec<f64>,
    sonra: Vec<f64>,
}

#[allow(clippy::too_many_lines)]
fn katman_ileri(
    spec: Spec,
    p: &Parametreler,
    l: usize,
    x: &[f64],
    kaynak: &[u32],
) -> (Vec<f64>, KatmanBellek) {
    let d = spec.d_model;
    let t = x.len() / d;
    let (ln1, o1, r1) = layer_norm_ileri(
        x,
        d,
        t,
        &p.ln1_olcek[l * d..(l + 1) * d],
        &p.ln1_sapma[l * d..(l + 1) * d],
    );
    let q = matmul(
        &ln1,
        &p.wq[l * d * d..(l + 1) * d * d],
        &p.bq[l * d..(l + 1) * d],
        d,
        d,
        t,
    );
    // Anahtarlar ve degerler KV genisliginde uretilir: sorgu genisligi
    // d_model, KV genisligi d_kv. Tam dikkatte ikisi esittir.
    let kv = spec.d_kv();
    let k = matmul(
        &ln1,
        &p.wk[l * d * kv..(l + 1) * d * kv],
        &p.bk[l * kv..(l + 1) * kv],
        d,
        kv,
        t,
    );
    let v = matmul(
        &ln1,
        &p.wv[l * d * kv..(l + 1) * d * kv],
        &p.bv[l * kv..(l + 1) * kv],
        d,
        kv,
        t,
    );
    let (attn, agirlik) = dikkat_ileri(spec, &q, &k, &v, t, kaynak);
    let cikti = matmul(
        &attn,
        &p.wo[l * d * d..(l + 1) * d * d],
        &p.bo[l * d..(l + 1) * d],
        d,
        d,
        t,
    );
    let kalinti1: Vec<f64> = x.iter().zip(cikti.iter()).map(|(a, b)| a + b).collect();
    let (ln2, o2, r2) = layer_norm_ileri(
        &kalinti1,
        d,
        t,
        &p.ln2_olcek[l * d..(l + 1) * d],
        &p.ln2_sapma[l * d..(l + 1) * d],
    );
    let on = matmul(
        &ln2,
        &p.w1[l * d * spec.d_ff..(l + 1) * d * spec.d_ff],
        &p.b1[l * spec.d_ff..(l + 1) * spec.d_ff],
        d,
        spec.d_ff,
        t,
    );
    let sonra: Vec<f64> = on.iter().map(|z| gelu(*z)).collect();
    let mlp = matmul(
        &sonra,
        &p.w2[l * spec.d_ff * d..(l + 1) * spec.d_ff * d],
        &p.b2[l * d..(l + 1) * d],
        spec.d_ff,
        d,
        t,
    );
    let y: Vec<f64> = kalinti1
        .iter()
        .zip(mlp.iter())
        .map(|(a, b)| a + b)
        .collect();
    (
        y,
        KatmanBellek {
            girdi: x.to_vec(),
            ln1,
            ortalama1: o1,
            rstd1: r1,
            q,
            k,
            v,
            agirlik,
            attn,
            kalinti1,
            ln2,
            ortalama2: o2,
            rstd2: r2,
            on,
            sonra,
        },
    )
}

#[allow(clippy::too_many_lines)]
fn katman_geri(
    spec: Spec,
    p: &Parametreler,
    grad: &mut Parametreler,
    l: usize,
    c: &KatmanBellek,
    dy: &[f64],
    kaynak: &[u32],
) -> Vec<f64> {
    let d = spec.d_model;
    let t = dy.len() / d;
    let f = spec.d_ff;

    // Residual: the MLP branch and the identity both receive dy.
    let dmlp = dy;
    let dsonra = matmul_t(dmlp, &p.w2[l * f * d..(l + 1) * f * d], f, d, t);
    for i in 0..t {
        for j in 0..d {
            grad.b2[l * d + j] += dy[i * d + j];
        }
    }
    for i in 0..t {
        for j in 0..f {
            for m in 0..d {
                grad.w2[l * f * d + m * f + j] += dmlp[i * d + m] * c.sonra[i * f + j];
            }
        }
    }
    let don: Vec<f64> = dsonra
        .iter()
        .zip(c.on.iter())
        .map(|(g, z)| g * gelu_turev(*z))
        .collect();
    let dln2 = matmul_t(&don, &p.w1[l * d * f..(l + 1) * d * f], d, f, t);
    for i in 0..t {
        for j in 0..f {
            grad.b1[l * f + j] += don[i * f + j];
        }
    }
    for i in 0..t {
        for j in 0..f {
            for m in 0..d {
                grad.w1[l * d * f + j * d + m] += don[i * f + j] * c.ln2[i * d + m];
            }
        }
    }
    let (dkalinti1, dg2, db2) = layer_norm_geri(
        &dln2,
        &c.ln2,
        &c.kalinti1,
        d,
        t,
        &c.ortalama2,
        &c.rstd2,
        &p.ln2_olcek[l * d..(l + 1) * d],
    );
    for i in 0..d {
        grad.ln2_olcek[l * d + i] += dg2[i];
        grad.ln2_sapma[l * d + i] += db2[i];
    }
    let dkalinti1: Vec<f64> = dkalinti1
        .iter()
        .zip(dy.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Attention output projection.
    let dattn = matmul_t(&dkalinti1, &p.wo[l * d * d..(l + 1) * d * d], d, d, t);
    for i in 0..t {
        for j in 0..d {
            grad.bo[l * d + j] += dkalinti1[i * d + j];
        }
    }
    for i in 0..t {
        for j in 0..d {
            for m in 0..d {
                grad.wo[l * d * d + j * d + m] += dkalinti1[i * d + j] * c.attn[i * d + m];
            }
        }
    }
    let (dq, dk, dv) = dikkat_geri(spec, &dattn, c, t, kaynak);

    // Q/K/V projections. K ve V gradyanlari KV genisligindedir; katkiyi ln1
    // uzayina tasiyan transpoz da ayni genislikle kurulur.
    let kv = spec.d_kv();
    let dln1 = matmul_t(&dq, &p.wq[l * d * d..(l + 1) * d * d], d, d, t);
    let dk_katkisi = matmul_t(&dk, &p.wk[l * d * kv..(l + 1) * d * kv], d, kv, t);
    let dv_katkisi = matmul_t(&dv, &p.wv[l * d * kv..(l + 1) * d * kv], d, kv, t);
    for i in 0..t {
        for j in 0..d {
            grad.bq[l * d + j] += dq[i * d + j];
        }
        for j in 0..kv {
            grad.bk[l * kv + j] += dk[i * kv + j];
            grad.bv[l * kv + j] += dv[i * kv + j];
        }
    }
    for i in 0..t {
        for j in 0..d {
            for m in 0..d {
                grad.wq[l * d * d + j * d + m] += dq[i * d + j] * c.ln1[i * d + m];
            }
        }
        for j in 0..kv {
            for m in 0..d {
                grad.wk[l * d * kv + j * d + m] += dk[i * kv + j] * c.ln1[i * d + m];
                grad.wv[l * d * kv + j * d + m] += dv[i * kv + j] * c.ln1[i * d + m];
            }
        }
    }
    let mut dln1_toplam = vec![0.0f64; t * d];
    for (toplam, parca) in dln1_toplam.iter_mut().zip(
        dln1.iter()
            .zip(dk_katkisi.iter())
            .map(|(a, b)| a + b)
            .zip(dv_katkisi.iter())
            .map(|(ab, c2)| ab + c2),
    ) {
        *toplam = parca;
    }
    let (dx_ln, dg1, db1) = layer_norm_geri(
        &dln1_toplam,
        &c.ln1,
        &c.girdi,
        d,
        t,
        &c.ortalama1,
        &c.rstd1,
        &p.ln1_olcek[l * d..(l + 1) * d],
    );
    for i in 0..d {
        grad.ln1_olcek[l * d + i] += dg1[i];
        grad.ln1_sapma[l * d + i] += db1[i];
    }
    dx_ln
        .iter()
        .zip(dkalinti1.iter())
        .map(|(a, b)| a + b)
        .collect()
}

/// `y[t] = W x[t] + b`, with `W` row-major `[cikti * girdi]`.
fn matmul(x: &[f64], w: &[f64], b: &[f64], girdi: usize, cikti: usize, t: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; t * cikti];
    for i in 0..t {
        for o in 0..cikti {
            let mut toplam = b[o];
            for m in 0..girdi {
                toplam += w[o * girdi + m] * x[i * girdi + m];
            }
            y[i * cikti + o] = toplam;
        }
    }
    y
}

/// `dx[t] = W^T dy[t]`, the transpose of [`matmul`] without the bias.
fn matmul_t(dy: &[f64], w: &[f64], girdi: usize, cikti: usize, t: usize) -> Vec<f64> {
    let mut dx = vec![0.0f64; t * girdi];
    for i in 0..t {
        for m in 0..girdi {
            let mut toplam = 0.0;
            for o in 0..cikti {
                toplam += w[o * girdi + m] * dy[i * cikti + o];
            }
            dx[i * girdi + m] = toplam;
        }
    }
    dx
}

fn layer_norm_ileri(
    x: &[f64],
    d: usize,
    t: usize,
    olcek: &[f64],
    sapma: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0f64; t * d];
    let mut ortalama = vec![0.0f64; t];
    let mut rstd = vec![0.0f64; t];
    for i in 0..t {
        let mut toplam = 0.0;
        for j in 0..d {
            toplam += x[i * d + j];
        }
        let ort = toplam / (d as f64);
        let mut varyans = 0.0;
        for j in 0..d {
            let fark = x[i * d + j] - ort;
            varyans += fark * fark;
        }
        varyans /= d as f64;
        let r = 1.0 / (varyans + LN_EPS).sqrt();
        ortalama[i] = ort;
        rstd[i] = r;
        for j in 0..d {
            y[i * d + j] = (x[i * d + j] - ort) * r * olcek[j] + sapma[j];
        }
    }
    (y, ortalama, rstd)
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_geri(
    dy: &[f64],
    _y: &[f64],
    x: &[f64],
    d: usize,
    t: usize,
    ortalama: &[f64],
    rstd: &[f64],
    olcek: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dx = vec![0.0f64; t * d];
    let mut dg = vec![0.0f64; d];
    let mut db = vec![0.0f64; d];
    let dn = d as f64;
    for i in 0..t {
        let r = rstd[i];
        let ort = ortalama[i];
        let mut xhat = vec![0.0f64; d];
        let mut dy_olcek = vec![0.0f64; d];
        for j in 0..d {
            xhat[j] = (x[i * d + j] - ort) * r;
            dy_olcek[j] = dy[i * d + j] * olcek[j];
            dg[j] += dy[i * d + j] * xhat[j];
            db[j] += dy[i * d + j];
        }
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for j in 0..d {
            s1 += dy_olcek[j];
            s2 += dy_olcek[j] * xhat[j];
        }
        for j in 0..d {
            dx[i * d + j] = r * (dy_olcek[j] - s1 / dn - xhat[j] * s2 / dn);
        }
    }
    (dx, dg, db)
}

/// Causal grouped-query attention forward; returns the concatenated heads and
/// the per-query-head weights, because backward needs them.
///
/// Query head `h` reads key/value head `h / grup`, where
/// `grup = n_heads / n_kv_heads` is how many query heads share one key/value
/// pair. With `grup == 1` (`kvh == head`, `kvd == d`) every index below is
/// the full-attention index, so the default spec runs the same arithmetic,
/// operation for operation.
fn dikkat_ileri(
    spec: Spec,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    t: usize,
    kaynak: &[u32],
) -> (Vec<f64>, Vec<f64>) {
    let d = spec.d_model;
    let h = spec.n_heads;
    let dk = spec.d_k();
    let kvd = spec.d_kv();
    let grup = h / spec.n_kv_heads;
    let mut cikti = vec![0.0f64; t * d];
    let mut agirliklar = vec![0.0f64; h * t * t];
    let olcek = 1.0 / (dk as f64).sqrt();
    for head in 0..h {
        let kvh = head / grup;
        for i in 0..t {
            let mut skor = vec![f64::NEG_INFINITY; t];
            for j in 0..=i {
                // Paketli pencerede kayit siniri asilmaz: baska bir kaydin
                // jetonuna bakmak, modelin alinti yapamayacagi bir baglam
                // kurmasi demektir.
                if kaynak[j] != kaynak[i] {
                    continue;
                }
                let mut toplam = 0.0;
                for m in 0..dk {
                    toplam += q[i * d + head * dk + m] * k[j * kvd + kvh * dk + m];
                }
                skor[j] = toplam * olcek;
            }
            let yumusak = softmax(&skor);
            for j in 0..t {
                agirliklar[head * t * t + i * t + j] = yumusak[j];
            }
            for m in 0..dk {
                let mut toplam = 0.0;
                for j in 0..t {
                    toplam += yumusak[j] * v[j * kvd + kvh * dk + m];
                }
                cikti[i * d + head * dk + m] = toplam;
            }
        }
    }
    (cikti, agirliklar)
}

/// Backward of [`dikkat_ileri`].
///
/// `dk` and `dv` live in the key/value width, so the gradient of a shared key
/// or value is the sum over the query heads of its group - accumulated in
/// ascending head order, because "which order" is a number the two kernels
/// have to agree on. With `grup == 1` every index is the full-attention index.
fn dikkat_geri(
    spec: Spec,
    dattn: &[f64],
    c: &KatmanBellek,
    t: usize,
    kaynak: &[u32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = spec.d_model;
    let h = spec.n_heads;
    let dk = spec.d_k();
    let kvd = spec.d_kv();
    let grup = h / spec.n_kv_heads;
    let mut dq = vec![0.0f64; t * d];
    let mut dkd = vec![0.0f64; t * kvd];
    let mut dv = vec![0.0f64; t * kvd];
    let olcek = 1.0 / (dk as f64).sqrt();
    for head in 0..h {
        let kvh = head / grup;
        for i in 0..t {
            // dv += w_ij * dout ; dw_ij = dout . v_j
            let mut dw = vec![0.0f64; t];
            for m in 0..dk {
                let g = dattn[i * d + head * dk + m];
                for (j, dw_deger) in dw.iter_mut().enumerate().take(i + 1) {
                    if kaynak[j] != kaynak[i] {
                        continue;
                    }
                    *dw_deger += g * c.v[j * kvd + kvh * dk + m];
                }
            }
            for m in 0..dk {
                let g = dattn[i * d + head * dk + m];
                for j in 0..=i {
                    if kaynak[j] != kaynak[i] {
                        continue;
                    }
                    let w = c.agirlik[head * t * t + i * t + j];
                    dv[j * kvd + kvh * dk + m] += g * w;
                }
            }
            // softmax backward over the causal row.
            let mut ds = vec![0.0f64; t];
            let mut dot = 0.0;
            let satir = &c.agirlik[head * t * t + i * t..head * t * t + (i + 1) * t];
            for (w, dw_deger) in satir.iter().zip(dw.iter()) {
                dot += w * dw_deger;
            }
            for ((j, ds_deger), dw_deger) in ds.iter_mut().enumerate().take(i + 1).zip(dw.iter()) {
                if kaynak[j] != kaynak[i] {
                    continue;
                }
                let w = c.agirlik[head * t * t + i * t + j];
                *ds_deger = w * (*dw_deger - dot);
            }
            for j in 0..=i {
                if kaynak[j] != kaynak[i] {
                    continue;
                }
                for m in 0..dk {
                    dq[i * d + head * dk + m] += ds[j] * olcek * c.k[j * kvd + kvh * dk + m];
                    dkd[j * kvd + kvh * dk + m] += ds[j] * olcek * c.q[i * d + head * dk + m];
                }
            }
        }
    }
    (dq, dkd, dv)
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let en_buyuk = x
        .iter()
        .fold(f64::NEG_INFINITY, |a, b| if *b > a { *b } else { a });
    let mut toplam = 0.0;
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
    for z in &y {
        toplam += z;
    }
    for z in &mut y {
        *z /= toplam;
    }
    y
}

fn softmax_ce(logits: &[f64], hedef: usize) -> (f64, Vec<f64>) {
    let olasilik = softmax(logits);
    let kayip = -olasilik[hedef].ln();
    let mut d = olasilik;
    d[hedef] -= 1.0;
    (kayip, d)
}

/// GELU, tanh form. The tanh form is used rather than the `erf` form because
/// its derivative is a closed form of exactly this function, so the gradient
/// check below compares like with like instead of comparing an analytic
/// derivative of one function against a numeric derivative of an approximation
/// of another.
fn gelu(z: f64) -> f64 {
    0.5 * z * (1.0 + gelu_ic(z))
}

/// `tanh(sqrt(2/pi) (z + 0.044715 z^3))`, the inner term of the tanh GELU.
fn gelu_ic(z: f64) -> f64 {
    let ic = (2.0 / std::f64::consts::PI).sqrt() * (z + 0.044_715 * z * z * z);
    ic.tanh()
}

/// The exact derivative of [`gelu`] as written above.
fn gelu_turev(z: f64) -> f64 {
    let t = gelu_ic(z);
    let dt = (2.0 / std::f64::consts::PI).sqrt() * (1.0 + 3.0 * 0.044_715 * z * z) * (1.0 - t * t);
    0.5 * (1.0 + t) + 0.5 * z * dt
}

/// The loss, forward only: no backward pass, no gradient allocation.
///
/// Validation asks one question - "how surprised is the model by text it did
/// not train on" - and the full pass answers it while building every cache the
/// backward pass needs and then throwing them away. This walks the same forward
/// route (same layer order, same mask, same tied readout and the same 1/d
/// scale, the same `toplam / t`) so its number is **bit-identical** to the one
/// the training path reports; `kayip_ileri_ileri_ve_geri_ile_ayni_sayiyi_verir`
/// asserts exactly that. If the two paths ever drift apart, that test fails.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn kayip_ileri(
    spec: Spec,
    p: &Parametreler,
    girdi: &[usize],
    hedef: &[usize],
    kaynak: &[u32],
) -> f64 {
    assert_eq!(
        kaynak.len(),
        girdi.len(),
        "kaynak vektoru girdiyle ayni uzunlukta olmali"
    );
    let d = spec.d_model;
    let t = girdi.len();
    let mut x = vec![0.0f64; t * d];
    for (i, tok) in girdi.iter().enumerate() {
        x[i * d..(i + 1) * d].copy_from_slice(&p.embedding[tok * d..(tok + 1) * d]);
    }
    for l in 0..spec.n_layers {
        let (y, _) = katman_ileri(spec, p, l, &x, kaynak);
        x = y;
    }
    let (xn, _, _) = layer_norm_ileri(&x, d, t, &p.lnf_olcek, &p.lnf_sapma);
    let olcek = 1.0 / (d as f64);
    let mut toplam_kayip = 0.0;
    for i in 0..t {
        let mut logits = vec![0.0f64; spec.vocab];
        let xn_satir = &xn[i * d..(i + 1) * d];
        for (v, logit) in logits.iter_mut().enumerate() {
            let satir = &p.embedding[v * d..(v + 1) * d];
            *logit = satir.iter().zip(xn_satir).map(|(a, b)| a * b).sum::<f64>() * olcek;
        }
        toplam_kayip += softmax_ce(&logits, hedef[i]).0;
    }
    toplam_kayip / (t as f64)
}

/// One epoch budget check: the ceiling belongs to the grant crate.
///
/// # Errors
/// A string naming the refusal; a zero epoch count and anything above
/// [`MAX_TRAINING_GRANT_EPOCHS`] are both refused.
pub fn epoch_butcesi(istenen: u32) -> Result<u32, String> {
    if istenen == 0 {
        return Err("0 epoch: kosulacak bir sey yok".to_string());
    }
    if istenen > MAX_TRAINING_GRANT_EPOCHS {
        return Err(format!(
            "{istenen} epoch protokol tavanini asiyor ({MAX_TRAINING_GRANT_EPOCHS})"
        ));
    }
    Ok(istenen)
}

/// AdamW state.
#[derive(Debug, Clone, PartialEq)]
pub struct Adamw {
    /// Step count.
    pub adim: u64,
    /// Learning rate.
    pub ogrenme_orani: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Numerical floor under the second-moment root.
    pub epsilon: f64,
    /// Weight decay, applied to the hidden weights only: decaying the embedding
    /// or a LayerNorm scale is not regularisation, it is shrinkage of a scale.
    pub agirlik_sonumu: f64,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adamw {
    /// # Errors
    /// A non-finite or non-positive learning rate, or a decay outside `[0, 1)`.
    pub fn yeni(olcu: usize, ogrenme_orani: f64, agirlik_sonumu: f64) -> Result<Self, String> {
        if !ogrenme_orani.is_finite() || ogrenme_orani <= 0.0 {
            return Err(format!("ogrenme orani {ogrenme_orani} gecersiz"));
        }
        if !(0.0..1.0).contains(&agirlik_sonumu) {
            return Err(format!("agirlik sonumu {agirlik_sonumu} [0,1) disinda"));
        }
        Ok(Self {
            adim: 0,
            ogrenme_orani,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            agirlik_sonumu,
            m: vec![0.0; olcu],
            v: vec![0.0; olcu],
        })
    }

    /// One decoupled-weight-decay update over one flat parameter block.
    ///
    /// `sonumlu` decides whether this block is decayed. It is a parameter of the
    /// call rather than a property of the optimiser because the answer differs
    /// per tensor: decaying a LayerNorm scale or the tied embedding shrinks a
    /// scale the model needs, it does not regularise anything.
    ///
    /// # Errors
    /// A length mismatch between the block, its gradient and the moment state.
    pub fn adim(&mut self, w: &mut [f64], gradyan: &[f64], sonumlu: bool) -> Result<(), String> {
        if w.len() != gradyan.len() || w.len() != self.m.len() {
            return Err(format!(
                "blok {} gradyan {} durum {} eleman: ayni tensore bakmiyorlar",
                w.len(),
                gradyan.len(),
                self.m.len()
            ));
        }
        self.adim += 1;
        let Self {
            adim,
            ogrenme_orani,
            beta1,
            beta2,
            epsilon,
            agirlik_sonumu,
            m,
            v,
        } = self;
        let duzeltme1 = 1.0 - beta1.powi(*adim as i32);
        let duzeltme2 = 1.0 - beta2.powi(*adim as i32);
        for (((w_deger, m_deger), v_deger), g) in w
            .iter_mut()
            .zip(m.iter_mut())
            .zip(v.iter_mut())
            .zip(gradyan)
        {
            *m_deger = *beta1 * *m_deger + (1.0 - *beta1) * *g;
            *v_deger = *beta2 * *v_deger + (1.0 - *beta2) * *g * *g;
            let m_hat = *m_deger / duzeltme1;
            let v_hat = *v_deger / duzeltme2;
            let mut delta = *ogrenme_orani * m_hat / (v_hat.sqrt() + *epsilon);
            if sonumlu {
                delta += *ogrenme_orani * *agirlik_sonumu * *w_deger;
            }
            *w_deger -= delta;
        }
        Ok(())
    }
}

/// What the corpus looks like to a fixed-window trainer, measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PencereRaporu {
    /// Records measured.
    pub kayit: usize,
    /// Tokens in total.
    pub toplam_jeton: usize,
    /// Full windows of the requested length.
    pub pencere: usize,
    /// Tokens that fall in a window.
    pub kapsanan_jeton: usize,
    /// Tokens left over in tails too short to fill a window.
    pub artan_jeton: usize,
    /// Windows if the corpus is packed into one stream instead of windowed
    /// record by record.
    pub paket_pencere: usize,
    /// Tokens left over when packing.
    pub paket_artan: usize,
    /// Median record length in tokens.
    pub p50: usize,
    /// 95th percentile record length in tokens.
    pub p95: usize,
    /// 99th percentile record length in tokens.
    pub p99: usize,
    /// Longest record, in tokens.
    pub en_uzun: usize,
}

/// Why a window measurement was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PencereHatasi {
    /// A zero-length window would produce no training signal.
    SifirUzunluk,
    /// No records at all.
    BosKorpus,
}

/// Measure the corpus against a fixed window length.
///
/// The window length comes from the spec's `max_seq_len`, and the point of
/// measuring it here is to test that number against the corpus the model will
/// actually train on rather than the one the spec was written from.
///
/// # Errors
/// [`PencereHatasi::SifirUzunluk`] on a zero window,
/// [`PencereHatasi::BosKorpus`] on no records.
pub fn pencere_olcu(
    jeton_sayilari: &[usize],
    uzunluk: usize,
) -> Result<PencereRaporu, PencereHatasi> {
    if uzunluk == 0 {
        return Err(PencereHatasi::SifirUzunluk);
    }
    if jeton_sayilari.is_empty() {
        return Err(PencereHatasi::BosKorpus);
    }
    let mut sirali = jeton_sayilari.to_vec();
    sirali.sort_unstable();
    let toplam: usize = sirali.iter().sum();
    let pencere: usize = sirali.iter().map(|n| n / uzunluk).sum();
    // En yakin-rank (nearest-rank) yontemi: rank = ceil(p * n), indeks rank - 1.
    // Yontem adıyla yaziliyor cunku "p95" tek basina bir sayi degil: (n-1)*p
    // yuvarlamasi ayni veride bir farkli deger verir ve iki yontem de "p95"
    // diye okunur. Karisiklik olmamasi icin secilen yontem burada duruyor.
    let yuzdelik = |p: f64| -> usize {
        let rank = (p * sirali.len() as f64).ceil() as usize;
        sirali[rank.saturating_sub(1).min(sirali.len() - 1)]
    };
    Ok(PencereRaporu {
        kayit: jeton_sayilari.len(),
        toplam_jeton: toplam,
        pencere,
        kapsanan_jeton: pencere * uzunluk,
        artan_jeton: toplam - pencere * uzunluk,
        paket_pencere: toplam / uzunluk,
        paket_artan: toplam % uzunluk,
        p50: yuzdelik(0.50),
        p95: yuzdelik(0.95),
        p99: yuzdelik(0.99),
        en_uzun: *sirali.last().unwrap_or(&0),
    })
}

/// One packed window: token ids plus where each one came from.
///
/// Packing is what makes the corpus usable - record-by-record windowing throws
/// away 81% of the tokens here - but a window that spans two records joins two
/// sources, and a citation is worthless if the passage it points at cannot be
/// attributed. So the provenance travels with the tokens instead of being
/// inferred later: `kaynak[i]` is the index of the record `kimlikler[i]` came
/// from, and the two arrays are the same length by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaketPencere {
    /// Token ids, in stream order.
    pub kimlikler: Vec<u32>,
    /// Source record index per position; same length as `kimlikler`.
    pub kaynak: Vec<u32>,
}

/// What packing did to the corpus, measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaketRaporu {
    /// Full windows produced.
    pub pencere: usize,
    /// Tokens that made it into a window.
    pub kapsanan_jeton: usize,
    /// Tokens left in the tail.
    pub artan_jeton: usize,
    /// Windows whose tokens all come from one record.
    pub tek_kaynakli: usize,
    /// Windows that join two or more records.
    pub cok_kaynakli: usize,
    /// The most records any single window joins.
    pub en_cok_kaynak: usize,
}

/// Why packing was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaketHatasi {
    /// A zero-length window carries no signal.
    SifirUzunluk,
    /// No records to pack.
    BosKorpus,
}

/// Pack records into one stream and cut fixed windows, keeping provenance.
///
/// # Errors
/// [`PaketHatasi::SifirUzunluk`] on a zero window,
/// [`PaketHatasi::BosKorpus`] on no records.
pub fn paketle(
    kayitlar: &[Vec<u32>],
    uzunluk: usize,
) -> Result<(Vec<PaketPencere>, PaketRaporu), PaketHatasi> {
    if uzunluk == 0 {
        return Err(PaketHatasi::SifirUzunluk);
    }
    if kayitlar.is_empty() {
        return Err(PaketHatasi::BosKorpus);
    }
    let toplam: usize = kayitlar.iter().map(Vec::len).sum();
    let mut akis_kimlik: Vec<u32> = Vec::with_capacity(toplam);
    let mut akis_kaynak: Vec<u32> = Vec::with_capacity(toplam);
    for (indeks, kayit) in kayitlar.iter().enumerate() {
        akis_kimlik.extend(kayit.iter().copied());
        akis_kaynak.extend(std::iter::repeat_n(indeks as u32, kayit.len()));
    }
    let tam = toplam / uzunluk;
    let mut pencereler = Vec::with_capacity(tam);
    let (mut tek, mut cok, mut en_cok) = (0usize, 0usize, 0usize);
    for w in 0..tam {
        let bas = w * uzunluk;
        let kaynak = akis_kaynak[bas..bas + uzunluk].to_vec();
        let mut ayrik: Vec<u32> = kaynak.clone();
        ayrik.sort_unstable();
        ayrik.dedup();
        if ayrik.len() > 1 {
            cok += 1;
        } else {
            tek += 1;
        }
        en_cok = en_cok.max(ayrik.len());
        pencereler.push(PaketPencere {
            kimlikler: akis_kimlik[bas..bas + uzunluk].to_vec(),
            kaynak,
        });
    }
    Ok((
        pencereler,
        PaketRaporu {
            pencere: tam,
            kapsanan_jeton: tam * uzunluk,
            artan_jeton: toplam - tam * uzunluk,
            tek_kaynakli: tek,
            cok_kaynakli: cok,
            en_cok_kaynak: en_cok,
        },
    ))
}

/// The embedding initialisation the μP table asks for: width-independent, so
/// the tied readout's `1/d_model` scale does not have to be re-tuned per width.
pub const INIT_STD_EMBEDDING: f64 = 1.0;

/// The parameter blocks, in the order a checkpoint stores them.
///
/// One list, used by three things: the checkpoint format (block names must
/// match on both sides), the weight-decay mask (which tensors are decayed), and
/// the shape check. Three copies of this list would be three chances to
/// disagree about what "the model" is.
pub const BLOK_ADLARI: [&str; 19] = [
    "embedding",
    "ln1_olcek",
    "ln1_sapma",
    "wq",
    "bq",
    "wk",
    "bk",
    "wv",
    "bv",
    "wo",
    "bo",
    "ln2_olcek",
    "ln2_sapma",
    "w1",
    "b1",
    "w2",
    "b2",
    "lnf_olcek",
    "lnf_sapma",
];

/// A deterministic normal stream: xorshift64 for bits, Box-Muller for shape.
///
/// Written out rather than pulled in, because a run that cannot be reproduced
/// from its seed is not a run anyone can compare a later run against, and the
/// only thing this has to be is the same on every machine.
struct Normal {
    durum: u64,
}

impl Normal {
    fn yeni(tohum: u64) -> Self {
        Self {
            durum: tohum
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        }
    }

    fn sonraki(&mut self) -> u64 {
        self.durum ^= self.durum << 13;
        self.durum ^= self.durum >> 7;
        self.durum ^= self.durum << 17;
        self.durum
    }

    /// Uniform in `(0, 1]`: the open end is at zero, because `ln(0)` is not a
    /// number and a Box-Muller pair built on one is a NaN weight.
    fn tek_duz(&mut self) -> f64 {
        let ham = (self.sonraki() >> 11) as f64;
        1.0 - ham / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.tek_duz();
        let u2 = self.tek_duz();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

impl Parametreler {
    /// Every weight at zero, in the spec's shape.
    #[must_use]
    pub fn sifir(spec: Spec) -> Self {
        let d = spec.d_model;
        let kv = spec.d_kv();
        let katman = spec.n_layers;
        Self {
            embedding: vec![0.0; spec.vocab * d],
            ln1_olcek: vec![1.0; katman * d],
            ln1_sapma: vec![0.0; katman * d],
            wq: vec![0.0; katman * d * d],
            bq: vec![0.0; katman * d],
            wk: vec![0.0; katman * d * kv],
            bk: vec![0.0; katman * kv],
            wv: vec![0.0; katman * d * kv],
            bv: vec![0.0; katman * kv],
            wo: vec![0.0; katman * d * d],
            bo: vec![0.0; katman * d],
            ln2_olcek: vec![1.0; katman * d],
            ln2_sapma: vec![0.0; katman * d],
            w1: vec![0.0; katman * spec.d_ff * d],
            b1: vec![0.0; katman * spec.d_ff],
            w2: vec![0.0; katman * d * spec.d_ff],
            b2: vec![0.0; katman * d],
            lnf_olcek: vec![1.0; d],
            lnf_sapma: vec![0.0; d],
        }
    }

    /// The μP initialisation the spec's parameter-group table describes.
    ///
    /// * hidden weights (`wq`/`wk`/`wv`/`wo`/`w1`/`w2`): `sqrt(2/fan_in)`,
    /// * the tied embedding: `base_init_std`, width-independent,
    /// * LayerNorm scales at one, every bias at zero.
    #[must_use]
    pub fn mup_init(spec: Spec, tohum: u64, base_init_std: f64) -> Self {
        let mut p = Self::sifir(spec);
        let mut akis = Normal::yeni(tohum);
        let d = spec.d_model;
        let f = spec.d_ff;
        let gizli = (2.0 / d as f64).sqrt();
        let yukari = (2.0 / d as f64).sqrt();
        let asagi = (2.0 / f as f64).sqrt();
        for deger in &mut p.embedding {
            *deger = akis.normal() * base_init_std;
        }
        for blok in [&mut p.wq, &mut p.wk, &mut p.wv, &mut p.wo] {
            for deger in blok.iter_mut() {
                *deger = akis.normal() * gizli;
            }
        }
        for deger in &mut p.w1 {
            *deger = akis.normal() * yukari;
        }
        for deger in &mut p.w2 {
            *deger = akis.normal() * asagi;
        }
        p
    }

    /// The blocks in the format's order.
    #[must_use]
    pub fn bloklar(&self) -> [&[f64]; 19] {
        [
            &self.embedding,
            &self.ln1_olcek,
            &self.ln1_sapma,
            &self.wq,
            &self.bq,
            &self.wk,
            &self.bk,
            &self.wv,
            &self.bv,
            &self.wo,
            &self.bo,
            &self.ln2_olcek,
            &self.ln2_sapma,
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.lnf_olcek,
            &self.lnf_sapma,
        ]
    }

    /// The blocks, mutable, in the same order.
    #[must_use]
    pub fn bloklar_mut(&mut self) -> [&mut [f64]; 19] {
        [
            &mut self.embedding,
            &mut self.ln1_olcek,
            &mut self.ln1_sapma,
            &mut self.wq,
            &mut self.bq,
            &mut self.wk,
            &mut self.bk,
            &mut self.wv,
            &mut self.bv,
            &mut self.wo,
            &mut self.bo,
            &mut self.ln2_olcek,
            &mut self.ln2_sapma,
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.lnf_olcek,
            &mut self.lnf_sapma,
        ]
    }

    /// The blocks with their format names, for a checkpoint that has to say
    /// which tensor each row belongs to.
    #[must_use]
    pub fn bloklar_adli(&self) -> Vec<(&'static str, &[f64])> {
        BLOK_ADLARI.iter().copied().zip(self.bloklar()).collect()
    }

    /// The format's block names.
    #[must_use]
    pub fn blok_adlari() -> Vec<&'static str> {
        BLOK_ADLARI.to_vec()
    }

    /// Put a named block back. `false` names a block that is not in the format.
    pub fn blok_ata(&mut self, ad: &str, degerler: Vec<f64>) -> bool {
        let Some(konum) = BLOK_ADLARI.iter().position(|a| *a == ad) else {
            return false;
        };
        let bloklar = self.bloklar_mut();
        if bloklar[konum].len() != degerler.len() {
            return false;
        }
        bloklar[konum].copy_from_slice(&degerler);
        true
    }

    /// How many numbers the whole model holds: the length an optimiser's moment
    /// vectors must have.
    #[must_use]
    pub fn toplam_ogeler(&self) -> usize {
        self.bloklar().iter().map(|b| b.len()).sum()
    }

    /// Whether every block is the length the spec's shape implies.
    #[must_use]
    pub fn sekil_dogru(&self, spec: Spec) -> bool {
        let d = spec.d_model;
        let kv = spec.d_kv();
        let katman = spec.n_layers;
        self.embedding.len() == spec.vocab * d
            && self.ln1_olcek.len() == katman * d
            && self.ln1_sapma.len() == katman * d
            && self.wq.len() == katman * d * d
            && self.bq.len() == katman * d
            && self.wk.len() == katman * d * kv
            && self.bk.len() == katman * kv
            && self.wv.len() == katman * d * kv
            && self.bv.len() == katman * kv
            && self.wo.len() == katman * d * d
            && self.bo.len() == katman * d
            && self.ln2_olcek.len() == katman * d
            && self.ln2_sapma.len() == katman * d
            && self.w1.len() == katman * spec.d_ff * d
            && self.b1.len() == katman * spec.d_ff
            && self.w2.len() == katman * d * spec.d_ff
            && self.b2.len() == katman * d
            && self.lnf_olcek.len() == d
            && self.lnf_sapma.len() == d
    }

    /// Add another parameter set position by position, block by block.
    pub fn topla_ile(&mut self, digeri: &Self) {
        let diger_bloklar = digeri.bloklar();
        for (hedef, kaynak) in self.bloklar_mut().iter_mut().zip(diger_bloklar) {
            if hedef.len() != kaynak.len() {
                continue;
            }
            for (a, b) in hedef.iter_mut().zip(kaynak) {
                *a += *b;
            }
        }
    }

    /// Scale every weight.
    pub fn olcekle(&mut self, k: f64) {
        for blok in self.bloklar_mut() {
            for deger in blok.iter_mut() {
                *deger *= k;
            }
        }
    }

    /// Which weights the optimiser decays, one flag per element.
    ///
    /// Decaying a LayerNorm scale or the tied embedding shrinks a scale the
    /// model needs; it does not regularise anything. The mask is the format's
    /// order, so it lines up with the moment vectors by construction.
    #[must_use]
    pub fn sonum_maskesi(&self) -> Vec<bool> {
        let mut maske = Vec::with_capacity(self.toplam_ogeler());
        for (ad, blok) in self.bloklar_adli() {
            let sonumlu = matches!(ad, "wq" | "wk" | "wv" | "wo" | "w1" | "w2");
            maske.extend(std::iter::repeat_n(sonumlu, blok.len()));
        }
        maske
    }

    /// Round every weight to `f32` and back.
    ///
    /// In place and irreversible on purpose: a checkpoint written at `f32`
    /// precision is a different model from the one that was trained, and the
    /// caller is expected to say so rather than to keep both.
    pub fn yuvarla_f32(&mut self) {
        for blok in self.bloklar_mut() {
            for deger in blok.iter_mut() {
                *deger = f64::from(*deger as f32);
            }
        }
    }
}

impl Adamw {
    /// Step count and the two moment vectors, for a checkpoint.
    #[must_use]
    pub fn durum(&self) -> (u64, &[f64], &[f64]) {
        (self.adim, &self.m, &self.v)
    }

    /// Rebuild an optimiser where it stopped.
    ///
    /// The step count matters as much as the weights: the bias correction is
    /// `1 - beta^step`, so a resumed run that starts the count at zero applies a
    /// different correction than the run it is continuing, and the loss curve
    /// moves for a reason that has nothing to do with the data.
    ///
    /// # Errors
    /// A length mismatch between the two moment vectors, or a hyper-parameter
    /// outside its range.
    pub fn durumdan(
        m: Vec<f64>,
        v: Vec<f64>,
        adim: u64,
        ogrenme_orani: f64,
        agirlik_sonumu: f64,
    ) -> Result<Self, String> {
        if m.len() != v.len() {
            return Err(format!(
                "moment vectors {} and {}: ayni modelin durumu degil",
                m.len(),
                v.len()
            ));
        }
        let mut o = Self::yeni(m.len(), ogrenme_orani, agirlik_sonumu)?;
        o.adim = adim;
        o.m = m;
        o.v = v;
        Ok(o)
    }

    /// One update over the whole model, with the decay decided per tensor.
    ///
    /// # Errors
    /// When the mask, the gradients or the moment vectors do not cover the same
    /// parameter, which is the only way this can be wrong silently.
    pub fn adim_maskele(
        &mut self,
        w: &mut Parametreler,
        gradyan: &Parametreler,
        maske: &[bool],
    ) -> Result<(), String> {
        let oge = w.toplam_ogeler();
        if maske.len() != oge {
            return Err(format!(
                "sonum maskesi {} ama parametre {oge}: maske modeli kaplamiyor",
                maske.len()
            ));
        }
        if gradyan.toplam_ogeler() != oge {
            return Err(format!(
                "gradyan {} parametre {oge}: ayni modele bakmiyorlar",
                gradyan.toplam_ogeler()
            ));
        }
        let Self {
            adim,
            ogrenme_orani,
            beta1,
            beta2,
            epsilon,
            agirlik_sonumu,
            m,
            v,
        } = self;
        *adim += 1;
        let duzeltme1 = 1.0 - beta1.powi(*adim as i32);
        let duzeltme2 = 1.0 - beta2.powi(*adim as i32);
        let (b1, b2, eps, oran, sonum) =
            (*beta1, *beta2, *epsilon, *ogrenme_orani, *agirlik_sonumu);
        let gradyanlar = gradyan.bloklar();
        let mut konum = 0usize;
        for (blok, grad) in w.bloklar_mut().iter_mut().zip(gradyanlar) {
            if blok.len() != grad.len() {
                return Err(format!(
                    "blok {} gradyan {}: ayni tensore bakmiyorlar",
                    blok.len(),
                    grad.len()
                ));
            }
            for (deger, g) in blok.iter_mut().zip(grad) {
                let mm = b1 * m[konum] + (1.0 - b1) * *g;
                let vv = b2 * v[konum] + (1.0 - b2) * *g * *g;
                m[konum] = mm;
                v[konum] = vv;
                let m_hat = mm / duzeltme1;
                let v_hat = vv / duzeltme2;
                let mut delta = oran * m_hat / (v_hat.sqrt() + eps);
                if maske[konum] {
                    delta += oran * sonum * *deger;
                }
                *deger -= delta;
                konum += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_packed_window_behaves_as_its_records_run_alone() {
        // Karar verici olcum: maske varsa, paketli kosunun kaybi parcalarin
        // tek basina kosularinin uzunlukla agirliklandirilmis ortalamasina
        // ESIT olur. Maske olmasa ikinci parca birincinin baglamini gorur ve
        // bu esitlik bozulur - yani test maskenin varligini olcuyor.
        let spec = kucuk_spec();
        let p = Parametreler::belirgin_doldur(spec, 23);
        let a_girdi = vec![0usize, 3, 1];
        let a_hedef = vec![3usize, 1, 6];
        let b_girdi = vec![6usize, 2, 5, 0];
        let b_hedef = vec![2usize, 5, 0, 4];

        let (kayip_a, _) = ileri_ve_geri(spec, &p, &a_girdi, &a_hedef);
        let (kayip_b, _) = ileri_ve_geri(spec, &p, &b_girdi, &b_hedef);

        let mut girdi = a_girdi.clone();
        girdi.extend_from_slice(&b_girdi);
        let mut hedef = a_hedef.clone();
        hedef.extend_from_slice(&b_hedef);
        let mut kaynak = vec![0u32; a_girdi.len()];
        kaynak.extend(std::iter::repeat_n(1u32, b_girdi.len()));

        let (kayip_paket, _) = ileri_ve_geri_paket(spec, &p, &girdi, &hedef, &kaynak);
        let beklenen =
            (kayip_a * a_girdi.len() as f64 + kayip_b * b_girdi.len() as f64) / girdi.len() as f64;
        assert!(
            (kayip_paket - beklenen).abs() < 1e-12,
            "paketli kayip {kayip_paket:.12} ama parcalar {beklenen:.12} diyor: \
             sinir maskesi calismiyor"
        );

        // Gradyan da ayni sekilde toplanir: paketli kosunun embedding gradyani,
        // iki tekil kosunun gradyanlarinin ortalamasi olmali.
        let (_, grad_a) = ileri_ve_geri(spec, &p, &a_girdi, &a_hedef);
        let (_, grad_b) = ileri_ve_geri(spec, &p, &b_girdi, &b_hedef);
        let (_, grad_paket) = ileri_ve_geri_paket(spec, &p, &girdi, &hedef, &kaynak);
        let na = a_girdi.len() as f64;
        let nb = b_girdi.len() as f64;
        for i in 0..grad_paket.embedding.len() {
            let beklenen_g = (grad_a.embedding[i] * na + grad_b.embedding[i] * nb) / (na + nb);
            assert!(
                (grad_paket.embedding[i] - beklenen_g).abs() < 1e-12,
                "embedding gradyani [{i}] sinir maskesiyle uyusmuyor"
            );
        }
    }

    #[test]
    fn a_mismatched_provenance_vector_is_refused() {
        let spec = kucuk_spec();
        let p = Parametreler::belirgin_doldur(spec, 5);
        let sonuc = std::panic::catch_unwind(|| {
            ileri_ve_geri_paket(spec, &p, &[0, 1, 2], &[1, 2, 3], &[0, 0])
        });
        assert!(
            sonuc.is_err(),
            "kaynak vektoru girdiden kisaydi kabul edildi"
        );
    }

    #[test]
    fn packing_keeps_provenance_next_to_every_token() {
        let kayitlar = vec![vec![10u32, 11, 12], vec![20, 21], vec![30]];
        let (pencereler, rapor) = paketle(&kayitlar, 4).unwrap();
        assert_eq!(rapor.pencere, 1);
        assert_eq!(rapor.kapsanan_jeton, 4);
        assert_eq!(rapor.artan_jeton, 2);
        assert_eq!(rapor.cok_kaynakli, 1);
        assert_eq!(rapor.en_cok_kaynak, 2);
        let p = &pencereler[0];
        assert_eq!(p.kimlikler, vec![10, 11, 12, 20]);
        assert_eq!(p.kaynak, vec![0, 0, 0, 1]);
        assert_eq!(p.kimlikler.len(), p.kaynak.len());
    }

    #[test]
    fn every_token_in_a_window_is_the_token_its_source_says() {
        // Provenance is the whole point of carrying it, so it is checked
        // against the records rather than trusted.
        let kayitlar = vec![vec![1u32, 2, 3, 4, 5], vec![6, 7, 8], vec![9]];
        let (pencereler, _) = paketle(&kayitlar, 3).unwrap();
        for pencere in &pencereler {
            for (kimlik, kaynak) in pencere.kimlikler.iter().zip(&pencere.kaynak) {
                assert!(
                    kayitlar[*kaynak as usize].contains(kimlik),
                    "jeton {kimlik} kaynak {kaynak} icinde yok"
                );
            }
        }
        // Ve sira korunuyor: pencereyi kaynaklara gore bolunce kayitlarin
        // kendisi cikmali.
        let birlesik: Vec<u32> = pencereler
            .iter()
            .flat_map(|p| p.kimlikler.clone())
            .collect();
        assert_eq!(birlesik, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn a_single_record_longer_than_the_window_is_not_counted_as_mixed() {
        let kayitlar = vec![vec![1u32, 2, 3, 4, 5, 6]];
        let (pencereler, rapor) = paketle(&kayitlar, 3).unwrap();
        assert_eq!(rapor.pencere, 2);
        assert_eq!(rapor.tek_kaynakli, 2);
        assert_eq!(rapor.cok_kaynakli, 0);
        assert_eq!(rapor.en_cok_kaynak, 1);
        assert_eq!(pencereler.len(), 2);
    }

    #[test]
    fn a_zero_window_and_an_empty_corpus_are_refused_by_packing() {
        assert_eq!(paketle(&[vec![1]], 0), Err(PaketHatasi::SifirUzunluk));
        assert_eq!(paketle(&[], 4), Err(PaketHatasi::BosKorpus));
    }

    #[test]
    fn windows_cover_only_whole_multiples_and_report_the_rest() {
        let rapor = pencere_olcu(&[10, 7, 25, 3], 8).unwrap();
        assert_eq!(rapor.kayit, 4);
        assert_eq!(rapor.toplam_jeton, 45);
        // Kayit basina: 10/8=1, 7/8=0, 25/8=3, 3/8=0 -> 4 pencere.
        assert_eq!(rapor.pencere, 4);
        assert_eq!(rapor.kapsanan_jeton, 32);
        assert_eq!(rapor.artan_jeton, 13);
        // Ayni korpus paketlenirse: 45/8 = 5 pencere, 5 jeton artik.
        assert_eq!(rapor.paket_pencere, 5);
        assert_eq!(rapor.paket_artan, 5);
        assert_eq!(rapor.p50, 7);
        assert_eq!(rapor.en_uzun, 25);
    }

    #[test]
    fn percentiles_are_measured_not_guessed() {
        let sayilar: Vec<usize> = (1..=100).collect();
        let rapor = pencere_olcu(&sayilar, 4).unwrap();
        assert_eq!(rapor.p50, 50);
        assert_eq!(rapor.p95, 95);
        assert_eq!(rapor.p99, 99);
        assert_eq!(rapor.en_uzun, 100);
    }

    #[test]
    fn a_zero_window_and_an_empty_corpus_are_refused() {
        assert_eq!(pencere_olcu(&[5], 0), Err(PencereHatasi::SifirUzunluk));
        assert_eq!(pencere_olcu(&[], 8), Err(PencereHatasi::BosKorpus));
    }

    use super::*;

    /// A configuration small enough that a finite-difference check over every
    /// parameter finishes quickly, and large enough to contain every code path:
    /// two layers, two heads, an MLP.
    fn kucuk_spec() -> Spec {
        Spec {
            vocab: 7,
            d_model: 4,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            d_ff: 6,
            max_seq_len: 16,
        }
    }

    /// Every parameter field, by name, so the check cannot quietly cover less
    /// than the model has.
    fn alanlar(p: &Parametreler) -> Vec<(&'static str, Vec<f64>)> {
        vec![
            ("embedding", p.embedding.clone()),
            ("wq", p.wq.clone()),
            ("wk", p.wk.clone()),
            ("wv", p.wv.clone()),
            ("wo", p.wo.clone()),
            ("w1", p.w1.clone()),
            ("w2", p.w2.clone()),
            ("bq", p.bq.clone()),
            ("bk", p.bk.clone()),
            ("bv", p.bv.clone()),
            ("bo", p.bo.clone()),
            ("b1", p.b1.clone()),
            ("b2", p.b2.clone()),
            ("ln1_olcek", p.ln1_olcek.clone()),
            ("ln1_sapma", p.ln1_sapma.clone()),
            ("ln2_olcek", p.ln2_olcek.clone()),
            ("ln2_sapma", p.ln2_sapma.clone()),
            ("lnf_olcek", p.lnf_olcek.clone()),
            ("lnf_sapma", p.lnf_sapma.clone()),
        ]
    }

    /// Write one delta into one field. Returns false for a name that does not
    /// exist, so a renamed field fails the check instead of being skipped.
    fn yaz(p: &mut Parametreler, ad: &str, i: usize, delta: f64) -> bool {
        let hedef: &mut Vec<f64> = match ad {
            "embedding" => &mut p.embedding,
            "wq" => &mut p.wq,
            "wk" => &mut p.wk,
            "wv" => &mut p.wv,
            "wo" => &mut p.wo,
            "w1" => &mut p.w1,
            "w2" => &mut p.w2,
            "bq" => &mut p.bq,
            "bk" => &mut p.bk,
            "bv" => &mut p.bv,
            "bo" => &mut p.bo,
            "b1" => &mut p.b1,
            "b2" => &mut p.b2,
            "ln1_olcek" => &mut p.ln1_olcek,
            "ln1_sapma" => &mut p.ln1_sapma,
            "ln2_olcek" => &mut p.ln2_olcek,
            "ln2_sapma" => &mut p.ln2_sapma,
            "lnf_olcek" => &mut p.lnf_olcek,
            "lnf_sapma" => &mut p.lnf_sapma,
            _ => return false,
        };
        hedef[i] += delta;
        true
    }

    /// Every analytic gradient against a central finite difference, over every
    /// parameter of the model.
    fn gradients_match_finite_differences(spec: Spec) -> Result<(), String> {
        spec.dogrula().map_err(|e| format!("spec refused: {e:?}"))?;
        let p = Parametreler::belirgin_doldur(spec, 7);
        // Jetonlar spec'in sozlugunden turetilir: ayni denetim kucuk ve
        // paylasimli spec'lerde de gecerli kalmali, taşiyan sabit değil.
        let girdi: Vec<usize> = (0..4).map(|i| (i * 2 + 1) % spec.vocab).collect();
        let hedef: Vec<usize> = (0..4).map(|i| (i * 2 + 2) % spec.vocab).collect();
        let (kayip0, grad) = ileri_ve_geri(spec, &p, &girdi, &hedef);
        if !kayip0.is_finite() || kayip0 <= 0.0 {
            return Err(format!("loss is not a usable number: {kayip0}"));
        }

        let h = 1e-5;
        let mut ihlaller: Vec<String> = Vec::new();
        let mut denetlenen = 0usize;
        for (ad, degerler) in alanlar(&p) {
            let analitik_alan = alanlar(&grad)
                .into_iter()
                .find(|(n, _)| *n == ad)
                .map_or_else(Vec::new, |(_, v)| v);
            if analitik_alan.len() != degerler.len() {
                return Err(format!("{ad}: gradient shape does not match the parameter"));
            }
            for (i, analitik_g) in analitik_alan.iter().enumerate() {
                let mut arti = p.clone();
                let mut eksi = p.clone();
                if !yaz(&mut arti, ad, i, h) || !yaz(&mut eksi, ad, i, -h) {
                    return Err(format!(
                        "field {ad} is not writable: the check would skip it"
                    ));
                }
                let (k1, _) = ileri_ve_geri(spec, &arti, &girdi, &hedef);
                let (k2, _) = ileri_ve_geri(spec, &eksi, &girdi, &hedef);
                let sonlu = (k1 - k2) / (2.0 * h);
                let fark = (analitik_g - sonlu).abs();
                let sinir = GRADIENT_CHECK_MUTLAK_TABAN
                    + GRADIENT_CHECK_TOLERANCE * analitik_g.abs().max(sonlu.abs());
                denetlenen += 1;
                if fark > sinir {
                    ihlaller.push(format!(
                        "{ad}[{i}] analitik={analitik_g:.6e} sonlu_fark={sonlu:.6e}"
                    ));
                }
            }
        }
        if !ihlaller.is_empty() {
            return Err(format!(
                "{} of {denetlenen} gradients disagree: {}",
                ihlaller.len(),
                ihlaller.join("; ")
            ));
        }
        // Tied to the spec's own count, not to a number written here: a field
        // added to the model without being added to the check then fails this
        // test instead of silently going unchecked.
        let beklenen = spec.parametre_sayisi();
        if denetlenen != beklenen {
            return Err(format!(
                "{denetlenen} gradients were checked but the spec has {beklenen} parameters"
            ));
        }
        Ok(())
    }

    #[test]
    fn backward_matches_finite_differences() {
        gradients_match_finite_differences(kucuk_spec()).unwrap();
    }

    /// Gruplanmis sorgu dikkatinde de her gradyan sonlu farkla uyusmali:
    /// KV paylasimi turevi degil indekslemeyi degistirir, bu yuzden ayni
    /// denetim paylasimli spec'te de gecmeli. Iki farkli grup orani
    /// denetlenir: 2:1 (iki sorgu basina bir KV basi) ve 4:2.
    #[test]
    fn gqa_gradyanlari_sonlu_farklarla_uyusur() {
        let iki_bir = Spec {
            n_kv_heads: 1,
            ..kucuk_spec()
        };
        gradients_match_finite_differences(iki_bir).unwrap();
        let dort_iki = Spec {
            vocab: 5,
            d_model: 8,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 8,
            max_seq_len: 8,
        };
        gradients_match_finite_differences(dort_iki).unwrap();
    }

    /// Varsayilan spec'in (n_kv_heads == n_heads) aritmetigi, KV paylasimini
    /// desteklemek icin dikkat yeniden yazilmasina ragmen bit duzeyinde ayni
    /// kalmali. Sabitler bu test yazilmadan onceki cekirdekle olculmustur;
    /// ikisinden biri degiserse varsayilan yol kaymis demektir. Kayip ve
    /// embedding gradyaninin ilk elemani yeter: ikisi de tum katmanlardan
    /// gecen birer akistir.
    #[test]
    fn tam_dikkat_yolu_bit_duzeyinde_korunur() {
        let spec = kucuk_spec();
        let p = Parametreler::belirgin_doldur(spec, 23);
        let girdi = vec![0usize, 3, 1, 6];
        let hedef = vec![3usize, 1, 6, 2];
        let (kayip, grad) = ileri_ve_geri(spec, &p, &girdi, &hedef);
        assert_eq!(
            kayip.to_bits(),
            0x3ffe_daf6_10c5_443e,
            "varsayilan yolun kaybi degisti"
        );
        assert_eq!(
            grad.embedding[0].to_bits(),
            0xbfa1_8256_dfa1_a4f9,
            "varsayilan yolun embedding gradyani degisti"
        );
    }

    /// GQA spec'inde parametre muhasebesi: dikkat artik 2d^2 (q, o) + 2dkv
    /// (k, v) + bias'lari sayar. Sayinin kendisi kapali formulle, farki ise
    /// tam dikkatten olan mesafeyle iki kez yazilir.
    #[test]
    fn gqa_parametre_muhasebesi() {
        let tam = Spec::lubot_a1();
        tam.dogrula().unwrap();
        assert_eq!(tam.parametre_sayisi(), 924_288);
        let paylasimli = Spec {
            n_kv_heads: 1,
            ..tam
        };
        paylasimli.dogrula().unwrap();
        // d_model 64, n_heads 2, n_kv_heads 1 -> d_k 32, d_kv 32.
        // Kapali form: 8192*64 + 8*(2*64^2 + 2*64*32 + 2*64 + 2*32)
        //            + 8*(2*64*256 + 256 + 64) + 8*4*64 + 2*64.
        assert_eq!(paylasimli.parametre_sayisi(), 891_008);
        let fark = tam.parametre_sayisi() - paylasimli.parametre_sayisi();
        let d = tam.d_model;
        let kv = paylasimli.d_kv();
        assert_eq!(fark, tam.n_layers * (2 * d * (d - kv) + 2 * (d - kv)));
    }

    /// KV bas sayisi bas sayisini bolmuyorsa spec reddedilmeli: 3 sorgu basina
    /// esit olmayan bir KV grubu sessizce yanlis indeks uzerinden okunurdu.
    #[test]
    fn bolunmeyen_kv_bas_sayisi_reddedilir() {
        let bozuk = Spec {
            n_kv_heads: 3,
            ..Spec::lubot_a1()
        };
        assert_eq!(bozuk.dogrula(), Err(SpecHatasi::KvBasSayisiBolunmuyor));
        let sifir = Spec {
            n_kv_heads: 0,
            ..Spec::lubot_a1()
        };
        assert_eq!(sifir.dogrula(), Err(SpecHatasi::KvBasSayisiBolunmuyor));
    }

    /// Paketli pencere maske kurali, KV paylasiminda da gecerli olmali:
    /// paylasimli spec'te paketli kayip, parcaların uzunlukla agirliklandir-
    /// ilmis ortalamasina esit olmali ve embedding gradyani da ayni sekilde
    /// toplanmali.
    #[test]
    fn gqa_paketli_pencere_kayit_sinirini_asmaz() {
        let spec = Spec {
            n_kv_heads: 1,
            ..kucuk_spec()
        };
        let p = Parametreler::belirgin_doldur(spec, 23);
        let a_girdi = vec![0usize, 3, 1];
        let a_hedef = vec![3usize, 1, 6];
        let b_girdi = vec![6usize, 2, 5, 0];
        let b_hedef = vec![2usize, 5, 0, 4];

        let (kayip_a, _) = ileri_ve_geri(spec, &p, &a_girdi, &a_hedef);
        let (kayip_b, _) = ileri_ve_geri(spec, &p, &b_girdi, &b_hedef);

        let mut girdi = a_girdi.clone();
        girdi.extend_from_slice(&b_girdi);
        let mut hedef = a_hedef.clone();
        hedef.extend_from_slice(&b_hedef);
        let mut kaynak = vec![0u32; a_girdi.len()];
        kaynak.extend(std::iter::repeat_n(1u32, b_girdi.len()));

        let (kayip_paket, _) = ileri_ve_geri_paket(spec, &p, &girdi, &hedef, &kaynak);
        let beklenen =
            (kayip_a * a_girdi.len() as f64 + kayip_b * b_girdi.len() as f64) / girdi.len() as f64;
        assert!(
            (kayip_paket - beklenen).abs() < 1e-12,
            "paylasimli dikkatte paketli kayip {kayip_paket:.12} ama parcalar {beklenen:.12}"
        );

        let (_, grad_a) = ileri_ve_geri(spec, &p, &a_girdi, &a_hedef);
        let (_, grad_b) = ileri_ve_geri(spec, &p, &b_girdi, &b_hedef);
        let (_, grad_paket) = ileri_ve_geri_paket(spec, &p, &girdi, &hedef, &kaynak);
        let na = a_girdi.len() as f64;
        let nb = b_girdi.len() as f64;
        for i in 0..grad_paket.embedding.len() {
            let beklenen_g = (grad_a.embedding[i] * na + grad_b.embedding[i] * nb) / (na + nb);
            assert!(
                (grad_paket.embedding[i] - beklenen_g).abs() < 1e-12,
                "paylasimli dikkatte embedding gradyani [{i}] sinir maskesiyle uyusmuyor"
            );
        }
    }

    #[test]
    fn the_spec_parameter_count_is_the_one_the_spec_claims() {
        let spec = Spec::lubot_a1();
        spec.dogrula().unwrap();
        assert_eq!(spec.parametre_sayisi(), 924_288);
        assert_eq!(spec.d_k(), 32);
        assert_eq!(spec.max_seq_len, 256);
    }

    #[test]
    fn a_spec_whose_heads_do_not_divide_is_refused() {
        let bozuk = Spec {
            n_heads: 3,
            ..Spec::lubot_a1()
        };
        assert_eq!(bozuk.dogrula(), Err(SpecHatasi::BasSayisiBolmuyor));
        let bos = Spec {
            d_ff: 0,
            ..Spec::lubot_a1()
        };
        assert_eq!(bos.dogrula(), Err(SpecHatasi::BosBoyut));
    }

    #[test]
    fn the_epoch_ceiling_comes_from_the_grant_crate() {
        assert_eq!(
            epoch_butcesi(0),
            Err("0 epoch: kosulacak bir sey yok".to_string())
        );
        assert_eq!(epoch_butcesi(4), Ok(4));
        assert!(epoch_butcesi(MAX_TRAINING_GRANT_EPOCHS + 1).is_err());
        assert_eq!(
            epoch_butcesi(MAX_TRAINING_GRANT_EPOCHS),
            Ok(MAX_TRAINING_GRANT_EPOCHS)
        );
    }

    #[test]
    fn gelu_and_its_derivative_agree_with_finite_differences() {
        for z in [-2.0, -0.5, 0.0, 0.3, 1.7] {
            let h = 1e-6;
            let sonlu = (gelu(z + h) - gelu(z - h)) / (2.0 * h);
            assert!(
                (gelu_turev(z) - sonlu).abs() < 1e-6,
                "gelu' at {z}: {} vs {sonlu}",
                gelu_turev(z)
            );
        }
    }

    #[test]
    fn a_short_descent_lowers_the_loss() {
        let spec = kucuk_spec();
        let mut p = Parametreler::belirgin_doldur(spec, 11);
        let girdi = vec![0usize, 3, 1, 6, 2];
        let hedef = vec![3usize, 1, 6, 2, 5];
        let (baslangic, _) = ileri_ve_geri(spec, &p, &girdi, &hedef);
        // One optimiser state per block: the moments belong to the tensor they
        // update, so a state shared across blocks would mix scales.
        let mut durumlar = [
            Adamw::yeni(p.embedding.len(), 0.05, 0.1).unwrap(),
            Adamw::yeni(p.wq.len(), 0.05, 0.1).unwrap(),
            Adamw::yeni(p.ln1_olcek.len(), 0.05, 0.1).unwrap(),
            Adamw::yeni(p.w1.len(), 0.05, 0.1).unwrap(),
            Adamw::yeni(p.lnf_olcek.len(), 0.05, 0.1).unwrap(),
        ];
        let mut son = baslangic;
        for _ in 0..40 {
            let (kayip, grad) = ileri_ve_geri(spec, &p, &girdi, &hedef);
            son = kayip;
            let bloklar: [(&mut [f64], &[f64], bool); 5] = [
                (&mut p.embedding[..], &grad.embedding[..], false),
                (&mut p.wq[..], &grad.wq[..], true),
                (&mut p.ln1_olcek[..], &grad.ln1_olcek[..], false),
                (&mut p.w1[..], &grad.w1[..], true),
                (&mut p.lnf_olcek[..], &grad.lnf_olcek[..], false),
            ];
            for (durum, (blok, gradyan, sonumlu)) in durumlar.iter_mut().zip(bloklar) {
                durum.adim(blok, gradyan, sonumlu).unwrap();
            }
        }
        assert!(son.is_finite(), "kayip sonlu degil: {son}");
        assert!(
            son < baslangic,
            "40 adim kaybi dusurmedi: {baslangic} -> {son}"
        );
    }

    #[test]
    fn an_optimizer_whose_state_does_not_match_the_block_is_refused() {
        let mut adamw = Adamw::yeni(4, 0.01, 0.1).unwrap();
        let mut w = vec![0.0f64; 4];
        assert!(adamw.adim(&mut w, &[0.0; 5], true).is_err());
        assert!(adamw.adim(&mut w, &[0.0; 4], true).is_ok());
    }

    #[test]
    fn a_degenerate_optimizer_is_refused() {
        assert!(Adamw::yeni(10, 0.0, 0.1).is_err());
        assert!(Adamw::yeni(10, f64::NAN, 0.1).is_err());
        assert!(Adamw::yeni(10, 1e-3, 1.0).is_err());
        assert!(Adamw::yeni(10, 1e-3, 0.1).is_ok());
    }

    /// Ileri-yalniz yol, egitim yolunun ayni sayisini vermeli: dogrulama
    /// egrisiyle egitim egrisi ayni seyi olcmeli, yoksa "dogrulama iyi"
    /// cumlesi baska bir sayidan gelir.
    #[test]
    fn kayip_ileri_ileri_ve_geri_ile_ayni_sayiyi_verir() {
        let spec = kucuk_spec();
        let p = Parametreler::belirgin_doldur(spec, 13);
        let kayit: Vec<usize> = (0..spec.max_seq_len)
            .map(|i| (i * 4 + 1) % spec.vocab)
            .collect();
        let n = kayit.len() - 1;
        let girdi = &kayit[..n];
        let hedef = &kayit[1..];
        let kaynak: Vec<u32> = (0..n).map(|i| u32::from(i >= n / 2)).collect();
        let sade = kayip_ileri(spec, &p, girdi, hedef, &kaynak);
        let (tam, _) = ileri_ve_geri_paket(spec, &p, girdi, hedef, &kaynak);
        assert_eq!(sade, tam, "ileri-yalniz yol ayristi: {sade} vs {tam}");
        // Geri gecis yok: ileri-yalniz yol hicbir sey degistirmemeli.
        let (tam2, _) = ileri_ve_geri_paket(spec, &p, girdi, hedef, &kaynak);
        assert_eq!(tam, tam2);
    }

    /// Dogrulamanin maliyeti, tam gecisin maliyetiyle olculur: ileri-yalniz
    /// yolun gerekcesi "daha ucuz" cumlesi degil, olculen orandir.
    #[test]
    #[ignore = "olcum: --ignored ile kosar"]
    #[allow(clippy::cast_precision_loss)]
    fn dogrulama_maliyeti_olcumu() {
        let spec = Spec::lubot_a1();
        let p = Parametreler::mup_init(spec, 5, INIT_STD_EMBEDDING);
        let kayit: Vec<usize> = (0..spec.max_seq_len)
            .map(|i| (i * 7 + 3) % spec.vocab)
            .collect();
        let n = kayit.len() - 1;
        let girdi = &kayit[..n];
        let hedef = &kayit[1..];
        let kaynak = vec![0u32; n];
        let mut tam = 0.0f64;
        let mut sade = 0.0f64;
        for _ in 0..2 {
            let s = std::time::Instant::now();
            let _ = ileri_ve_geri_paket(spec, &p, girdi, hedef, &kaynak);
            tam += s.elapsed().as_secs_f64();
            let s = std::time::Instant::now();
            let _ = kayip_ileri(spec, &p, girdi, hedef, &kaynak);
            sade += s.elapsed().as_secs_f64();
        }
        println!(
            "tam {:?} ms, ileri-yalniz {:?} ms, oran {:.2}x",
            tam / 2.0 * 1000.0,
            sade / 2.0 * 1000.0,
            tam / sade
        );
    }
}
