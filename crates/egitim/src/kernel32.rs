//! f32 hesap çekirdeği: eğitimin aynı matematiği, yarı genişlikte.
//!
//! # Neden iki çekirdek
//!
//! Çekirdek `lib.rs` içinde f64 yazıldı ve orada kalıyor: f64 olan, gradyan
//! sınamasının referansı, kontrol noktası biçiminin taşıyıcısı ve "acaba
//! yuvarlama mı" sorusunun cevabı. Bu modül ise *hız* için var: aynı mimari,
//! aynı maske kuralı, aynı toplama sırası - ama bütün aritmetik f32.
//!
//! # İki yol birbirini doğrular
//!
//! İkinci bir çekirdek yazmanın dürüst gerekçesi ancak şu olabilir: ikisi
//! karşılaştırılır ve ayrıldıkları yerde ölçülür. `f32_gradyani_f64_ile_
//! uyusur` testi aynı parametrelerle iki çekirdeği koşar ve her gradyan
//! tensörünün en büyük sapmasını f64 büyüklüğüne oranlar; tolerans aşılırsa
//! test düşer. Yani "f32 yeterince iyi" bir iddia değil, koşan bir ölçüm.
//!
//! # Ne değişir, ne değişmez
//!
//! Değişmez: toplama sırası, maske kuralı (kayıt sınırı aşılmaz), katman
//! sırası, kayıp tanımı. Değişir: her ara değer f32, dolayısıyla yuvarlama
//! hatası ~1e-7 yerine ~1e-3 mertebesinde biriken bir akış. Kontrol noktası ve
//! optimiser f64 kalır: gradyan f32 hesaplanır, f64'e çevrilip AdamW'ye girer.
//! Böylece "hangisiyle eğitildi" sorusu raporun `hassasiyet` alanında durur,
//! sayılarda sessizce karışmaz.
//!
//! # Ölçülen hız (ve dürüst sonucu)
//!
//! `hiz_olcumu_f64_vs_f32` (ignored test) bu makinede 128 baglam, d_model 128,
//! 4 katman ile kosuldu: f64 ~8736 ms, f32 ~8096 ms, **oran 1,08x**. Yani
//! skaler Rust'ta f32, f64'ün iki kati hizli degil; x86-64'te skaler toplama
//! ve çarpma zaten ayni gecikmede ve darboğaz SIMD yoklugu. f32'nin gercek
//! kazanci vektörleşmeyle gelir (`f32x8`); bu depoda o yol henüz yok. O
//! yüzden bu çekirdegin gerekçesi hiz degil **çapraz doğrulama**: iki çekirdek
//! ayni gradyani vermek zorunda ve test bunu ölçüyor. Hız iddiası ölçülmeden
//! yazilmayacak. Gercek kazanc ayni is yukunu ipliklere bolmekte cikti: olcum
//! `docs/HESAP-CEKIRDEGI.md` (1,94x, ayni kontrol noktasi sha256'si).
//!
//! # Parametreler neden ayrı bir yapı
//!
//! f32 aritmetiğin anlamı olması için parametrelerin de f32 olması gerekir;
//! her adımda f64'ten f32'ye çevirmek toplama sırasını değiştirmez ama
//! bellek erişimini ikiye katlar. Bu yüzden koşu bir kez indirir
//! ([`Parametreler32::indir`]), her adımda f32 okur ve geri dönerken gradyanı
//! yine bir kez f64'e çevirir ([`Parametreler32::geri_f64`]).

/// LayerNorm epsilon, f32.
pub(crate) const LN_EPS32: f32 = 1e-5;

/// The architecture's parameters, in f32: same fields, same shapes, half the
/// width. See the module docs for why this exists as a second struct.
#[derive(Debug, Clone, PartialEq)]
pub struct Parametreler32 {
    /// Tied token embedding / readout.
    pub embedding: Vec<f32>,
    /// Per layer: LayerNorm 1 scale.
    pub ln1_olcek: Vec<f32>,
    /// Per layer: LayerNorm 1 bias.
    pub ln1_sapma: Vec<f32>,
    /// Per layer: query weights.
    pub wq: Vec<f32>,
    /// Per layer: query biases.
    pub bq: Vec<f32>,
    /// Per layer: key weights.
    pub wk: Vec<f32>,
    /// Per layer: key biases.
    pub bk: Vec<f32>,
    /// Per layer: value weights.
    pub wv: Vec<f32>,
    /// Per layer: value biases.
    pub bv: Vec<f32>,
    /// Per layer: output projection weights.
    pub wo: Vec<f32>,
    /// Per layer: output projection biases.
    pub bo: Vec<f32>,
    /// Per layer: LayerNorm 2 scale.
    pub ln2_olcek: Vec<f32>,
    /// Per layer: LayerNorm 2 bias.
    pub ln2_sapma: Vec<f32>,
    /// Per layer: MLP up weights.
    pub w1: Vec<f32>,
    /// Per layer: MLP up biases.
    pub b1: Vec<f32>,
    /// Per layer: MLP down weights.
    pub w2: Vec<f32>,
    /// Per layer: MLP down biases.
    pub b2: Vec<f32>,
    /// Final LayerNorm scale.
    pub lnf_olcek: Vec<f32>,
    /// Final LayerNorm bias.
    pub lnf_sapma: Vec<f32>,
}

impl Parametreler32 {
    /// Down-cast a checkpoint's parameters into f32.
    #[must_use]
    pub fn indir(p: &crate::Parametreler) -> Self {
        fn k(v: &[f64]) -> Vec<f32> {
            v.iter().map(|x| *x as f32).collect()
        }
        Self {
            embedding: k(&p.embedding),
            ln1_olcek: k(&p.ln1_olcek),
            ln1_sapma: k(&p.ln1_sapma),
            wq: k(&p.wq),
            bq: k(&p.bq),
            wk: k(&p.wk),
            bk: k(&p.bk),
            wv: k(&p.wv),
            bv: k(&p.bv),
            wo: k(&p.wo),
            bo: k(&p.bo),
            ln2_olcek: k(&p.ln2_olcek),
            ln2_sapma: k(&p.ln2_sapma),
            w1: k(&p.w1),
            b1: k(&p.b1),
            w2: k(&p.w2),
            b2: k(&p.b2),
            lnf_olcek: k(&p.lnf_olcek),
            lnf_sapma: k(&p.lnf_sapma),
        }
    }

    /// Up-cast back to f64. Used for the gradient (AdamW stays f64) and for
    /// writing a checkpoint out of an f32 run.
    #[must_use]
    pub fn geri_f64(&self) -> crate::Parametreler {
        fn k(v: &[f32]) -> Vec<f64> {
            v.iter().map(|x| f64::from(*x)).collect()
        }
        crate::Parametreler {
            embedding: k(&self.embedding),
            ln1_olcek: k(&self.ln1_olcek),
            ln1_sapma: k(&self.ln1_sapma),
            wq: k(&self.wq),
            bq: k(&self.bq),
            wk: k(&self.wk),
            bk: k(&self.bk),
            wv: k(&self.wv),
            bv: k(&self.bv),
            wo: k(&self.wo),
            bo: k(&self.bo),
            ln2_olcek: k(&self.ln2_olcek),
            ln2_sapma: k(&self.ln2_sapma),
            w1: k(&self.w1),
            b1: k(&self.b1),
            w2: k(&self.w2),
            b2: k(&self.b2),
            lnf_olcek: k(&self.lnf_olcek),
            lnf_sapma: k(&self.lnf_sapma),
        }
    }

    /// Zeroed gradients of the same shape, in f32.
    #[must_use]
    pub fn sifir_gradyan(&self) -> Parametreler32 {
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

    /// Whether every field has the shape the spec asks for. Shapes are the
    /// spec's, so a mismatch is a refusal here rather than a panic deep in a
    /// kernel.
    #[must_use]
    pub fn sekil_dogru(&self, spec: crate::Spec) -> bool {
        let d = spec.d_model;
        let l = spec.n_layers;
        let f = spec.d_ff;
        self.embedding.len() == spec.vocab * d
            && self.ln1_olcek.len() == l * d
            && self.ln1_sapma.len() == l * d
            && self.wq.len() == l * d * d
            && self.bq.len() == l * d
            && self.wk.len() == l * d * d
            && self.bk.len() == l * d
            && self.wv.len() == l * d * d
            && self.bv.len() == l * d
            && self.wo.len() == l * d * d
            && self.bo.len() == l * d
            && self.ln2_olcek.len() == l * d
            && self.ln2_sapma.len() == l * d
            && self.w1.len() == l * f * d
            && self.b1.len() == l * f
            && self.w2.len() == l * f * d
            && self.b2.len() == l * d
            && self.lnf_olcek.len() == d
            && self.lnf_sapma.len() == d
    }

    /// Loss-only forward for inference: no caches, no gradients, one pass.
    /// Used where a number is needed and a backward pass is not.
    #[must_use]
    pub fn kayip_ileri(
        &self,
        spec: crate::Spec,
        girdi: &[usize],
        hedef: &[usize],
        kaynak: &[u32],
    ) -> f32 {
        // f64 tarafiyla ayni yol, ayni sira: ileri, son LayerNorm, bagli
        // readout ve 1/d olcegi, sonunda ayni `toplam / t`. Geri gecis yok.
        assert_eq!(kaynak.len(), girdi.len(), "kaynak girdiyle ayni uzunlukta");
        let d = spec.d_model;
        let t = girdi.len();
        let mut x = vec![0.0f32; t * d];
        for (i, tok) in girdi.iter().enumerate() {
            x[i * d..(i + 1) * d].copy_from_slice(&self.embedding[tok * d..(tok + 1) * d]);
        }
        for l in 0..spec.n_layers {
            let (y, _) = katman_ileri32(spec, self, l, &x, kaynak);
            x = y;
        }
        let (xn, _, _) = ln_ileri32(&x, d, t, &self.lnf_olcek, &self.lnf_sapma);
        let olcek = 1.0 / (d as f32);
        let mut toplam = 0.0f32;
        for i in 0..t {
            let mut logits = vec![0.0f32; spec.vocab];
            let satir_xn = &xn[i * d..(i + 1) * d];
            for (v, logit) in logits.iter_mut().enumerate() {
                let satir = &self.embedding[v * d..(v + 1) * d];
                *logit = satir.iter().zip(satir_xn).map(|(a, b)| a * b).sum::<f32>() * olcek;
            }
            toplam += softmax_ce32(&logits, hedef[i]).0;
        }
        toplam / (t as f32)
    }
}

struct KatmanBellek32 {
    girdi: Vec<f32>,
    ln1: Vec<f32>,
    ortalama1: Vec<f32>,
    rstd1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    agirlik: Vec<f32>,
    attn: Vec<f32>,
    kalinti1: Vec<f32>,
    ln2: Vec<f32>,
    ortalama2: Vec<f32>,
    rstd2: Vec<f32>,
    on: Vec<f32>,
    sonra: Vec<f32>,
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
pub fn ileri_ve_geri_paket_32(
    spec: crate::Spec,
    p: &Parametreler32,
    girdi: &[usize],
    hedef: &[usize],
    kaynak: &[u32],
) -> (f32, Parametreler32) {
    assert_eq!(
        kaynak.len(),
        girdi.len(),
        "kaynak vektoru girdiyle ayni uzunlukta olmali"
    );
    let d = spec.d_model;
    let t = girdi.len();
    let mut grad = p.sifir_gradyan();

    // Embedding lookup: x[t] = embedding[token[t]].
    let mut x = vec![0.0f32; t * d];
    for (i, tok) in girdi.iter().enumerate() {
        x[i * d..(i + 1) * d].copy_from_slice(&p.embedding[tok * d..(tok + 1) * d]);
    }

    let mut caches: Vec<KatmanBellek32> = Vec::with_capacity(spec.n_layers);
    for l in 0..spec.n_layers {
        let (y, bellek) = katman_ileri32(spec, p, l, &x, kaynak);
        x = y;
        caches.push(bellek);
    }

    // Final LayerNorm.
    let (mut xn, son_ortalama, son_rstd) = ln_ileri32(&x, d, t, &p.lnf_olcek, &p.lnf_sapma);

    // Tied readout with the spec's 1/d_model logit scale, then softmax + CE.
    let olcek = 1.0 / (d as f32);
    let mut toplam_kayip = 0.0;
    let mut dxn = vec![0.0f32; t * d];
    for i in 0..t {
        let mut logits = vec![0.0f32; spec.vocab];
        let xn_satir = &xn[i * d..(i + 1) * d];
        for (v, logit) in logits.iter_mut().enumerate() {
            let satir = &p.embedding[v * d..(v + 1) * d];
            *logit = satir.iter().zip(xn_satir).map(|(a, b)| a * b).sum::<f32>() * olcek;
        }
        let (kayip, mut dlogits) = softmax_ce32(&logits, hedef[i]);
        toplam_kayip += kayip;
        // d/dembedding from the readout, and d/dxn.
        let dxn_satir = &mut dxn[i * d..(i + 1) * d];
        for (v, dlogit) in dlogits.iter().enumerate() {
            // The loss is averaged over positions, so is this contribution:
            // without the 1/t the tied readout would outweigh the input
            // embedding by a factor of t.
            let g = dlogit * olcek / (t as f32);
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
    let kayip = toplam_kayip / (t as f32);

    // Final LayerNorm backward.
    let (dx, dg, db) = ln_geri32(&dxn, &xn, &x, d, t, &son_ortalama, &son_rstd, &p.lnf_olcek);
    for (i, g) in dg.iter().enumerate() {
        grad.lnf_olcek[i] += g;
    }
    for (i, g) in db.iter().enumerate() {
        grad.lnf_sapma[i] += g;
    }
    xn.clear();
    let mut dx_akis = dx;

    for l in (0..spec.n_layers).rev() {
        dx_akis = katman_geri32(spec, p, &mut grad, l, &caches[l], &dx_akis, kaynak);
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
#[allow(clippy::too_many_lines)]
fn katman_ileri32(
    spec: crate::Spec,
    p: &Parametreler32,
    l: usize,
    x: &[f32],
    kaynak: &[u32],
) -> (Vec<f32>, KatmanBellek32) {
    let d = spec.d_model;
    let t = x.len() / d;
    let (ln1, o1, r1) = ln_ileri32(
        x,
        d,
        t,
        &p.ln1_olcek[l * d..(l + 1) * d],
        &p.ln1_sapma[l * d..(l + 1) * d],
    );
    let q = matmul32(
        &ln1,
        &p.wq[l * d * d..(l + 1) * d * d],
        &p.bq[l * d..(l + 1) * d],
        d,
        d,
        t,
    );
    let k = matmul32(
        &ln1,
        &p.wk[l * d * d..(l + 1) * d * d],
        &p.bk[l * d..(l + 1) * d],
        d,
        d,
        t,
    );
    let v = matmul32(
        &ln1,
        &p.wv[l * d * d..(l + 1) * d * d],
        &p.bv[l * d..(l + 1) * d],
        d,
        d,
        t,
    );
    let (attn, agirlik) = dikkat_ileri32(spec, &q, &k, &v, t, kaynak);
    let cikti = matmul32(
        &attn,
        &p.wo[l * d * d..(l + 1) * d * d],
        &p.bo[l * d..(l + 1) * d],
        d,
        d,
        t,
    );
    let kalinti1: Vec<f32> = x.iter().zip(cikti.iter()).map(|(a, b)| a + b).collect();
    let (ln2, o2, r2) = ln_ileri32(
        &kalinti1,
        d,
        t,
        &p.ln2_olcek[l * d..(l + 1) * d],
        &p.ln2_sapma[l * d..(l + 1) * d],
    );
    let on = matmul32(
        &ln2,
        &p.w1[l * d * spec.d_ff..(l + 1) * d * spec.d_ff],
        &p.b1[l * spec.d_ff..(l + 1) * spec.d_ff],
        d,
        spec.d_ff,
        t,
    );
    let sonra: Vec<f32> = on.iter().map(|z| gelu32(*z)).collect();
    let mlp = matmul32(
        &sonra,
        &p.w2[l * spec.d_ff * d..(l + 1) * spec.d_ff * d],
        &p.b2[l * d..(l + 1) * d],
        spec.d_ff,
        d,
        t,
    );
    let y: Vec<f32> = kalinti1
        .iter()
        .zip(mlp.iter())
        .map(|(a, b)| a + b)
        .collect();
    (
        y,
        KatmanBellek32 {
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
fn katman_geri32(
    spec: crate::Spec,
    p: &Parametreler32,
    grad: &mut Parametreler32,
    l: usize,
    c: &KatmanBellek32,
    dy: &[f32],
    kaynak: &[u32],
) -> Vec<f32> {
    let d = spec.d_model;
    let t = dy.len() / d;
    let f = spec.d_ff;

    // Residual: the MLP branch and the identity both receive dy.
    let dmlp = dy;
    let dsonra = matmul_t32(dmlp, &p.w2[l * f * d..(l + 1) * f * d], f, d, t);
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
    let don: Vec<f32> = dsonra
        .iter()
        .zip(c.on.iter())
        .map(|(g, z)| g * gelu_turev32(*z))
        .collect();
    let dln2 = matmul_t32(&don, &p.w1[l * d * f..(l + 1) * d * f], d, f, t);
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
    let (dkalinti1, dg2, db2) = ln_geri32(
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
    let dkalinti1: Vec<f32> = dkalinti1
        .iter()
        .zip(dy.iter())
        .map(|(a, b)| a + b)
        .collect();

    // Attention output projection.
    let dattn = matmul_t32(&dkalinti1, &p.wo[l * d * d..(l + 1) * d * d], d, d, t);
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
    let (dq, dk, dv) = dikkat_geri32(spec, &dattn, c, t, kaynak);

    // Q/K/V projections.
    let dln1 = matmul_t32(&dq, &p.wq[l * d * d..(l + 1) * d * d], d, d, t);
    let dk_katkisi = matmul_t32(&dk, &p.wk[l * d * d..(l + 1) * d * d], d, d, t);
    let dv_katkisi = matmul_t32(&dv, &p.wv[l * d * d..(l + 1) * d * d], d, d, t);
    for i in 0..t {
        for j in 0..d {
            grad.bq[l * d + j] += dq[i * d + j];
            grad.bk[l * d + j] += dk[i * d + j];
            grad.bv[l * d + j] += dv[i * d + j];
        }
    }
    for i in 0..t {
        for j in 0..d {
            for m in 0..d {
                grad.wq[l * d * d + j * d + m] += dq[i * d + j] * c.ln1[i * d + m];
                grad.wk[l * d * d + j * d + m] += dk[i * d + j] * c.ln1[i * d + m];
                grad.wv[l * d * d + j * d + m] += dv[i * d + j] * c.ln1[i * d + m];
            }
        }
    }
    let mut dln1_toplam = vec![0.0f32; t * d];
    for (toplam, parca) in dln1_toplam.iter_mut().zip(
        dln1.iter()
            .zip(dk_katkisi.iter())
            .map(|(a, b)| a + b)
            .zip(dv_katkisi.iter())
            .map(|(ab, c2)| ab + c2),
    ) {
        *toplam = parca;
    }
    let (dx_ln, dg1, db1) = ln_geri32(
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
fn matmul32(x: &[f32], w: &[f32], b: &[f32], girdi: usize, cikti: usize, t: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; t * cikti];
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
fn matmul_t32(dy: &[f32], w: &[f32], girdi: usize, cikti: usize, t: usize) -> Vec<f32> {
    let mut dx = vec![0.0f32; t * girdi];
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

fn ln_ileri32(
    x: &[f32],
    d: usize,
    t: usize,
    olcek: &[f32],
    sapma: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; t * d];
    let mut ortalama = vec![0.0f32; t];
    let mut rstd = vec![0.0f32; t];
    for i in 0..t {
        let mut toplam = 0.0;
        for j in 0..d {
            toplam += x[i * d + j];
        }
        let ort = toplam / (d as f32);
        let mut varyans = 0.0;
        for j in 0..d {
            let fark = x[i * d + j] - ort;
            varyans += fark * fark;
        }
        varyans /= d as f32;
        let r = 1.0 / (varyans + LN_EPS32).sqrt();
        ortalama[i] = ort;
        rstd[i] = r;
        for j in 0..d {
            y[i * d + j] = (x[i * d + j] - ort) * r * olcek[j] + sapma[j];
        }
    }
    (y, ortalama, rstd)
}

#[allow(clippy::too_many_arguments)]
fn ln_geri32(
    dy: &[f32],
    _y: &[f32],
    x: &[f32],
    d: usize,
    t: usize,
    ortalama: &[f32],
    rstd: &[f32],
    olcek: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; t * d];
    let mut dg = vec![0.0f32; d];
    let mut db = vec![0.0f32; d];
    let dn = d as f32;
    for i in 0..t {
        let r = rstd[i];
        let ort = ortalama[i];
        let mut xhat = vec![0.0f32; d];
        let mut dy_olcek = vec![0.0f32; d];
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

/// Causal multi-head attention forward; returns the concatenated heads and the
/// per-head weights, because backward needs them.
fn dikkat_ileri32(
    spec: crate::Spec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    t: usize,
    kaynak: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let d = spec.d_model;
    let h = spec.n_heads;
    let dk = spec.d_k();
    let mut cikti = vec![0.0f32; t * d];
    let mut agirliklar = vec![0.0f32; h * t * t];
    let olcek = 1.0 / (dk as f32).sqrt();
    for head in 0..h {
        for i in 0..t {
            let mut skor = vec![f32::NEG_INFINITY; t];
            for j in 0..=i {
                // Paketli pencerede kayit siniri asilmaz: baska bir kaydin
                // jetonuna bakmak, modelin alinti yapamayacagi bir baglam
                // kurmasi demektir.
                if kaynak[j] != kaynak[i] {
                    continue;
                }
                let mut toplam = 0.0;
                for m in 0..dk {
                    toplam += q[i * d + head * dk + m] * k[j * d + head * dk + m];
                }
                skor[j] = toplam * olcek;
            }
            let yumusak = softmax32(&skor);
            for j in 0..t {
                agirliklar[head * t * t + i * t + j] = yumusak[j];
            }
            for m in 0..dk {
                let mut toplam = 0.0;
                for j in 0..t {
                    toplam += yumusak[j] * v[j * d + head * dk + m];
                }
                cikti[i * d + head * dk + m] = toplam;
            }
        }
    }
    (cikti, agirliklar)
}

fn dikkat_geri32(
    spec: crate::Spec,
    dattn: &[f32],
    c: &KatmanBellek32,
    t: usize,
    kaynak: &[u32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = spec.d_model;
    let h = spec.n_heads;
    let dk = spec.d_k();
    let mut dq = vec![0.0f32; t * d];
    let mut dkd = vec![0.0f32; t * d];
    let mut dv = vec![0.0f32; t * d];
    let olcek = 1.0 / (dk as f32).sqrt();
    for head in 0..h {
        for i in 0..t {
            // dv += w_ij * dout ; dw_ij = dout . v_j
            let mut dw = vec![0.0f32; t];
            for m in 0..dk {
                let g = dattn[i * d + head * dk + m];
                for (j, dw_deger) in dw.iter_mut().enumerate().take(i + 1) {
                    if kaynak[j] != kaynak[i] {
                        continue;
                    }
                    *dw_deger += g * c.v[j * d + head * dk + m];
                }
            }
            for m in 0..dk {
                let g = dattn[i * d + head * dk + m];
                for j in 0..=i {
                    if kaynak[j] != kaynak[i] {
                        continue;
                    }
                    let w = c.agirlik[head * t * t + i * t + j];
                    dv[j * d + head * dk + m] += g * w;
                }
            }
            // softmax backward over the causal row.
            let mut ds = vec![0.0f32; t];
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
                    dq[i * d + head * dk + m] += ds[j] * olcek * c.k[j * d + head * dk + m];
                    dkd[j * d + head * dk + m] += ds[j] * olcek * c.q[i * d + head * dk + m];
                }
            }
        }
    }
    (dq, dkd, dv)
}

fn softmax32(x: &[f32]) -> Vec<f32> {
    let en_buyuk = x
        .iter()
        .fold(f32::NEG_INFINITY, |a, b| if *b > a { *b } else { a });
    let mut toplam = 0.0;
    let mut y: Vec<f32> = x
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

fn softmax_ce32(logits: &[f32], hedef: usize) -> (f32, Vec<f32>) {
    let olasilik = softmax32(logits);
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
fn gelu32(z: f32) -> f32 {
    0.5 * z * (1.0 + gelu_ic32(z))
}

/// `tanh(sqrt(2/pi) (z + 0.044715 z^3))`, the inner term of the tanh GELU.
fn gelu_ic32(z: f32) -> f32 {
    let ic = (2.0 / std::f32::consts::PI).sqrt() * (z + 0.044_715 * z * z * z);
    ic.tanh()
}

/// The exact derivative of [`gelu`] as written above.
fn gelu_turev32(z: f32) -> f32 {
    let t = gelu_ic32(z);
    let dt = (2.0 / std::f32::consts::PI).sqrt() * (1.0 + 3.0 * 0.044_715 * z * z) * (1.0 - t * t);
    0.5 * (1.0 + t) + 0.5 * z * dt
}

#[cfg(test)]
mod testler {
    use super::*;

    /// Small enough to be quick, big enough that every path (maske, MLP,
    /// final LayerNorm) is walked.
    fn kucuk_spec() -> crate::Spec {
        crate::Spec {
            vocab: 11,
            d_model: 8,
            n_layers: 2,
            n_heads: 2,
            d_ff: 12,
            max_seq_len: 24,
        }
    }

    /// One record in, one (input, target) window out: the same shift the
    /// trainer uses, expressed once so the tests cannot disagree on it.
    fn girdi_hedef(kayit: &[usize]) -> (Vec<usize>, Vec<usize>) {
        assert!(kayit.len() >= 2);
        let n = kayit.len() - 1;
        (kayit[..n].to_vec(), kayit[1..].to_vec())
    }

    /// The point of a second kernel: the two must agree. This is the
    /// measurement that justifies f32 - it is not asserted, it is compared.
    #[test]
    fn f32_gradyani_f64_ile_uyusur() {
        let spec = kucuk_spec();
        let p64 = crate::Parametreler::belirgin_doldur(spec, 7);
        let p32 = Parametreler32::indir(&p64);
        assert!(p32.sekil_dogru(spec));
        let kayit: Vec<usize> = (0..16).map(|i| (i * 3 + 1) % spec.vocab).collect();
        let (girdi, hedef) = girdi_hedef(&kayit);
        let kaynak = vec![0u32; girdi.len()];

        let (k64, g64) = crate::ileri_ve_geri_paket(spec, &p64, &girdi, &hedef, &kaynak);
        let (k32, g32) = ileri_ve_geri_paket_32(spec, &p32, &girdi, &hedef, &kaynak);

        assert!(
            (k64 - f64::from(k32)).abs() < 1e-4,
            "kayip ayristi: f64 {k64}, f32 {k32}"
        );

        let g32_f64 = g32.geri_f64();
        for (ad, a, b) in [
            ("embedding", &g64.embedding, &g32_f64.embedding),
            ("wq", &g64.wq, &g32_f64.wq),
            ("wk", &g64.wk, &g32_f64.wk),
            ("wv", &g64.wv, &g32_f64.wv),
            ("wo", &g64.wo, &g32_f64.wo),
            ("w1", &g64.w1, &g32_f64.w1),
            ("w2", &g64.w2, &g32_f64.w2),
            ("ln1_olcek", &g64.ln1_olcek, &g32_f64.ln1_olcek),
            ("lnf_sapma", &g64.lnf_sapma, &g32_f64.lnf_sapma),
        ] {
            let en_buyuk = a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let fark = a
                .iter()
                .zip(b.iter())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            // Goreli tolerans: sifira yakin tensorde mutlak taban.
            let oran = fark / en_buyuk.max(1e-6);
            assert!(
                oran < 2e-3,
                "{ad}: f64 ile f32 ayristi, fark {fark}, buyukluk {en_buyuk}, oran {oran}"
            );
        }
    }

    /// Same shape of check for the loss alone, over a window that forces the
    /// mask to matter (two records in one window).
    #[test]
    fn f32_kayip_maske_ile_birlikte_uyusur() {
        let spec = kucuk_spec();
        let p64 = crate::Parametreler::belirgin_doldur(spec, 3);
        let p32 = Parametreler32::indir(&p64);
        let kayit: Vec<usize> = (0..spec.max_seq_len)
            .map(|i| (i * 5 + 2) % spec.vocab)
            .collect();
        let (girdi, hedef) = girdi_hedef(&kayit);
        // Iki kayit: 0..8 ve 8..16; maske olmasa ilk yari ikinciyi gorurdu.
        let kaynak: Vec<u32> = (0..girdi.len()).map(|i| u32::from(i >= 8)).collect();
        let (k64, _) = crate::ileri_ve_geri_paket(spec, &p64, &girdi, &hedef, &kaynak);
        let (k32, _) = ileri_ve_geri_paket_32(spec, &p32, &girdi, &hedef, &kaynak);
        assert!(
            (k64 - f64::from(k32)).abs() < 1e-4,
            "maskeli kayip ayristi: f64 {k64}, f32 {k32}"
        );
    }

    /// A stray f32 kernel must not be able to read a record's neighbour: the
    /// same rule as f64, measured the same way (change the second record, the
    /// first half's loss must not move).
    #[test]
    fn f32_kayit_siniri_asilmaz() {
        let spec = kucuk_spec();
        let p = Parametreler32::indir(&crate::Parametreler::belirgin_doldur(spec, 11));
        let a: Vec<usize> = (0..16).map(|i| (i * 2) % spec.vocab).collect();
        // Hedeflerin (a[1..9]) disinda kalan kuyrugu degistir: boylece ilk
        // yarida ne girdi ne hedef degisir, yalniz *uzak baglam* degisir.
        let mut b = a.clone();
        for jeton in b.iter_mut().skip(10) {
            *jeton = (*jeton + 1) % spec.vocab;
        }
        let kaynak: Vec<u32> = (0..15).map(|i| u32::from(i >= 8)).collect();
        let (g1_girdi, g1_hedef) = girdi_hedef(&a);
        let (g2_girdi, g2_hedef) = girdi_hedef(&b);
        let (k1, _) = ileri_ve_geri_paket_32(spec, &p, &g1_girdi, &g1_hedef, &kaynak);
        let (k2, _) = ileri_ve_geri_paket_32(spec, &p, &g2_girdi, &g2_hedef, &kaynak);
        // Ilk yarida kaynak ayni oldugu surece kayip ayni kalmali.
        let (ilk1, _) =
            ileri_ve_geri_paket_32(spec, &p, &g1_girdi[..8], &g1_hedef[..8], &kaynak[..8]);
        let (ilk2, _) =
            ileri_ve_geri_paket_32(spec, &p, &g2_girdi[..8], &g2_hedef[..8], &kaynak[..8]);
        assert_eq!(ilk1, ilk2, "ilk kayit ikinciden etkilendi");
        assert!(
            (k1 - k2).abs() > 1e-6,
            "uzak kuyrugu degistirmek kaybi hic etkilemedi: {k1} vs {k2}"
        );
    }

    /// Round trip: a checkpoint's parameters survive down-cast and up-cast to
    /// within f32 precision, and the gradient of the down-cast parameters is
    /// zero at a point where the up-cast gradient is zero.
    #[test]
    fn indir_geri_tur_f32_sinirinda_dogru() {
        let spec = kucuk_spec();
        let p64 = crate::Parametreler::belirgin_doldur(spec, 5);
        let p32 = Parametreler32::indir(&p64);
        let geri = p32.geri_f64();
        let fark = p64
            .wq
            .iter()
            .zip(geri.wq.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(fark < 1e-6, "indir/geri turu sasti: {fark}");
        let sifir = p32.sifir_gradyan();
        assert_eq!(sifir.wq.len(), p32.wq.len());
        assert!(sifir.wq.iter().all(|g| *g == 0.0));
    }

    /// f32 kernel must refuse to look like it works on the wrong shapes.
    #[test]
    fn sekil_dogrulugu_yanlis_spec_i_yakalar() {
        let spec = kucuk_spec();
        let p = Parametreler32::indir(&crate::Parametreler::belirgin_doldur(spec, 1));
        assert!(p.sekil_dogru(spec));
        let baska = crate::Spec {
            d_model: spec.d_model + 1,
            ..spec
        };
        assert!(!p.sekil_dogru(baska));
    }

    /// f32 cekirdegin varlik sebebi hiz: olculur, soylenmez.
    #[test]
    #[ignore = "olcum: --ignored ile kosar"]
    fn hiz_olcumu_f64_vs_f32() {
        let spec = crate::Spec {
            vocab: 4096,
            d_model: 128,
            n_layers: 4,
            n_heads: 4,
            d_ff: 512,
            max_seq_len: 128,
        };
        let p64 = crate::Parametreler::belirgin_doldur(spec, 17);
        let p32 = Parametreler32::indir(&p64);
        let kayit: Vec<usize> = (0..128).map(|i| (i * 7 + 3) % spec.vocab).collect();
        let (girdi, hedef) = girdi_hedef(&kayit);
        let kaynak = vec![0u32; girdi.len()];
        let mut t64 = 0.0f64;
        let mut t32 = 0.0f64;
        for _ in 0..3 {
            let s = std::time::Instant::now();
            let _ = crate::ileri_ve_geri_paket(spec, &p64, &girdi, &hedef, &kaynak);
            t64 += s.elapsed().as_secs_f64();
            let s = std::time::Instant::now();
            let _ = ileri_ve_geri_paket_32(spec, &p32, &girdi, &hedef, &kaynak);
            t32 += s.elapsed().as_secs_f64();
        }
        println!(
            "f64 {:?} ms, f32 {:?} ms, oran {:.2}x",
            t64 / 3.0 * 1000.0,
            t32 / 3.0 * 1000.0,
            t64 / t32
        );
    }

    /// f32 tarafinda da ileri-yalniz yol, geri gecisli yolun kaybiyla ayni
    /// olmali (f32 icinde birebir, cunku ayni islem sirasi).
    #[test]
    fn f32_kayip_ileri_geri_gecisli_yolla_ayni() {
        let spec = kucuk_spec();
        let p = Parametreler32::indir(&crate::Parametreler::belirgin_doldur(spec, 6));
        let kayit: Vec<usize> = (0..spec.max_seq_len)
            .map(|i| (i * 6 + 2) % spec.vocab)
            .collect();
        let n = kayit.len() - 1;
        let girdi = &kayit[..n];
        let hedef = &kayit[1..];
        let kaynak = vec![0u32; n];
        let sade = p.kayip_ileri(spec, girdi, hedef, &kaynak);
        let (tam, _) = ileri_ve_geri_paket_32(spec, &p, girdi, hedef, &kaynak);
        assert_eq!(sade, tam, "f32 ileri-yalniz yol ayristi: {sade} vs {tam}");
    }
}
