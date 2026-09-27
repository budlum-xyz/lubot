//! # hadamard - the Kronecker-Walsh conditioned feed-forward
//!
//! The port card for this module (kept in the work repository, with the
//! upstream licence note) describes a feed-forward block that replaces the
//! dense `d_model -> d_ff -> d_model` sandwich with three *structured*
//! transforms. The structure is the whole point: a dense layer of width `n`
//! costs `n^2` parameters, while a Kronecker pair of Walsh factors costs
//! `ba^2 + bb^2` with `ba * bb == n` - for `n = 1024` that is 2048 numbers
//! instead of 1048576, and `carp_tasarrufu` below computes the ratio as
//! integer arithmetic rather than claiming it.
//!
//! What this file ports is the *arithmetic*, not the text: the pipeline, the
//! initialisers and the constants are the card's, the names and the module
//! layout are this repository's. Nothing is imported, downloaded or vendored.
//!
//! The pipeline, in order:
//!
//! ```text
//! z  = pad(x, n)                       // n = next power of two >= d_model
//! z  = K(d1 * z)            then p1    // Walsh pair 1, fixed permutation
//! z  = K(silu(d2 * c * z + b2)) then p2 // Walsh pair 2, fixed permutation
//! z  = K(d3 * z)                       // Walsh pair 3
//! y  = (d4 * z)[..d_model]
//! ```
//!
//! `c` is the conditioning vector: `c = 1 + softmax(x W_v) W_u` with a rank of
//! [`KOSUL_RANK`]. `W_u` starts at zero, so a fresh block has `c == 1`
//! *exactly* - the conditioning is a no-op until it is trained, and that is a
//! bit-identity here rather than an approximation (`kosul_sifirken_birdir`).
//!
//! ## One declared divergence
//!
//! The two fixed permutations are drawn from this repository's own generator
//! ([`crate::Tohum`]) rather than from the card's generator. A permutation
//! chosen once and then frozen is not a numerical property of the transform -
//! any bijection interleaves the Kronecker factors the same way - but the
//! *specific* permutation differs, so two implementations will not agree bit
//! for bit on trained weights. This is written down rather than glossed:
//! `karisim_gercek_permutasyon` measures that what we produce is a bijection,
//! `karisim_belirlenimci` that it is the same one on every machine.
//!
//! ## What is not here
//!
//! No backward pass. This crate is forward-only, so this block can be
//! measured, counted and compared, but not trained; whether it replaces the
//! gated feed-forward in the backbone family is the marked decision M1 and is
//! not taken here.

use crate::Tohum;

/// The rank of the conditioning path.
pub const KOSUL_RANK: usize = 8;

/// Standard deviation of the conditioning projection's initialiser.
pub const KOSUL_INIT_STD: f32 = 0.02;

/// The output diagonal starts here, not at one: the block begins small.
pub const CIKIS_DIAGONAL_INIT: f32 = 0.02;

/// Seeds for the two frozen permutations.
///
/// Two constants rather than one: the second permutation must not be the
/// first one again, or the pipeline would fold two mixing steps into one.
pub const KARISIM_TOHUMLARI: (u64, u64) = (11, 13);

/// Offset for the upper half when the permutation may not cross the middle.
pub const KARISIM_UST_YARI_OFSETI: u64 = 977;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HadamardHatasi {
    /// A width of zero has no transform.
    SifirGenislik,
    /// The input length is not a multiple of the model width.
    GirdiSekli { beklenen_kati: usize, gelen: usize },
}

impl core::fmt::Display for HadamardHatasi {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SifirGenislik => write!(f, "width zero"),
            Self::GirdiSekli {
                beklenen_kati,
                gelen,
            } => write!(f, "input {gelen} is not a multiple of {beklenen_kati}"),
        }
    }
}

impl std::error::Error for HadamardHatasi {}

/// The normalised Walsh-Hadamard matrix of size `n`, row-major.
///
/// `n` must be a power of two. The entries are `+-1/sqrt(n)`, which makes the
/// matrix orthogonal *and* symmetric, so it is its own inverse - measured in
/// `walsh_kendi_tersi`, because an unnormalised Walsh matrix would grow the
/// signal by `sqrt(n)` at every one of the three stages.
#[must_use]
pub fn walsh(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let olcek = 1.0 / (n as f32).sqrt();
    let mut m = vec![0.0f32; n * n];
    for (i, satir) in m.chunks_exact_mut(n).enumerate() {
        for (j, hucre) in satir.iter_mut().enumerate() {
            // Sylvester's construction: the sign is the parity of the
            // population count of the bitwise and.
            let isaret = if (i & j).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            *hucre = isaret * olcek;
        }
    }
    m
}

/// Split `n` into the two Kronecker block widths, `(ba, bb)` with `ba * bb == n`.
///
/// The split is as square as a power of two allows, because `ba^2 + bb^2` is
/// smallest there: for `n = 1024` the split is `32 x 32` (2048 numbers), while
/// a lopsided `2 x 512` split would cost 262148.
#[must_use]
pub fn blok_bol(n: usize) -> (usize, usize) {
    if n <= 1 {
        return (1, n.max(1));
    }
    let bit_uzunlugu = usize::BITS - (n - 1).leading_zeros();
    let ba = 1usize << (bit_uzunlugu / 2);
    (ba, n / ba)
}

/// The Kronecker product applied without ever building the product.
///
/// `z` is read as a `ba x bb` matrix `Z`; the result is `A^T Z B` read back as
/// a vector. Building `kron(A, B)` first would be the same numbers at `n^2`
/// cost, which is what `kron_yogun_esdeger` measures.
#[must_use]
pub fn kron_uygula(z: &[f32], a: &[f32], ba: usize, b: &[f32], bb: usize) -> Vec<f32> {
    let mut ara = vec![0.0f32; ba * bb];
    // ara[i][l] = sum_j Z[i][j] B[j][l]
    for i in 0..ba {
        for j in 0..bb {
            let zij = z[i * bb + j];
            if zij == 0.0 {
                continue;
            }
            for l in 0..bb {
                ara[i * bb + l] += zij * b[j * bb + l];
            }
        }
    }
    let mut cikis = vec![0.0f32; ba * bb];
    // cikis[k][l] = sum_i A[i][k] ara[i][l]
    for i in 0..ba {
        for k in 0..ba {
            let aik = a[i * ba + k];
            if aik == 0.0 {
                continue;
            }
            for l in 0..bb {
                cikis[k * bb + l] += aik * ara[i * bb + l];
            }
        }
    }
    cikis
}

/// A frozen permutation of `0..n`, drawn once from `tohum`.
///
/// Fisher-Yates over this repository's generator. When `yariya_ayrik` is set
/// the two halves are shuffled separately and never mix, which is what a
/// half-split rotary pairing needs if the two halves are to keep meaning the
/// same thing.
#[must_use]
pub fn karisim(n: usize, tohum: u64, yariya_ayrik: bool) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    if yariya_ayrik && n % 2 == 0 {
        let yari = n / 2;
        let mut alt = karisim(yari, tohum, false);
        let ust = karisim(yari, tohum.wrapping_add(KARISIM_UST_YARI_OFSETI), false);
        alt.extend(ust.into_iter().map(|v| v + yari));
        return alt;
    }
    let mut p: Vec<usize> = (0..n).collect();
    let mut t = Tohum::yeni(tohum);
    for i in (1..n).rev() {
        let j = (t.tam_sayi() % ((i + 1) as u64)) as usize;
        p.swap(i, j);
    }
    p
}

/// `silu(x) = x * sigmoid(x)`, evaluated in double precision.
#[must_use]
pub fn silu(v: f32) -> f32 {
    let x = f64::from(v);
    (x / (1.0 + (-x).exp())) as f32
}

/// Softmax over one row, shifted by the maximum so a large score cannot
/// overflow before it is exponentiated.
#[must_use]
pub fn yumusak_azami(satir: &[f32]) -> Vec<f32> {
    if satir.is_empty() {
        return Vec::new();
    }
    let enb = satir.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut us: Vec<f32> = satir
        .iter()
        .map(|v| (f64::from(*v - enb)).exp() as f32)
        .collect();
    let toplam: f32 = us.iter().sum();
    if toplam > 0.0 {
        for v in &mut us {
            *v /= toplam;
        }
    }
    us
}

/// The block's shape, and the parameter count that follows from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HadamardSekli {
    pub d_model: usize,
    /// The padded width: the next power of two at or above `d_model`.
    pub n: usize,
    pub ba: usize,
    pub bb: usize,
}

impl HadamardSekli {
    /// # Errors
    ///
    /// [`HadamardHatasi::SifirGenislik`] when `d_model` is zero.
    pub fn yeni(d_model: usize) -> Result<Self, HadamardHatasi> {
        if d_model == 0 {
            return Err(HadamardHatasi::SifirGenislik);
        }
        let n = d_model.next_power_of_two();
        let (ba, bb) = blok_bol(n);
        Ok(Self { d_model, n, ba, bb })
    }

    /// Parameters, from the shape alone.
    ///
    /// Three Walsh pairs, five diagonals (`d1 d2 d3 d4` and the bias `b2`),
    /// and the two conditioning projections.
    #[must_use]
    pub fn param_sayisi(&self) -> usize {
        3 * (self.ba * self.ba + self.bb * self.bb)
            + 5 * self.n
            + KOSUL_RANK * (self.d_model + self.n)
    }

    /// What one dense `n x n` layer would have cost, for the same width.
    #[must_use]
    pub fn yogun_param_sayisi(&self) -> usize {
        self.n * self.n
    }

    /// The saving, as an exact integer ratio `(dense, structured)`.
    #[must_use]
    pub fn carp_tasarrufu(&self) -> (usize, usize) {
        (
            self.yogun_param_sayisi(),
            3 * (self.ba * self.ba + self.bb * self.bb),
        )
    }
}

/// The block itself: shape, frozen permutations and weights.
#[derive(Debug, Clone)]
pub struct Hadamard {
    sekil: HadamardSekli,
    /// Three `(A, B)` factor pairs, in pipeline order.
    faktorler: [(Vec<f32>, Vec<f32>); 3],
    d1: Vec<f32>,
    d2: Vec<f32>,
    b2: Vec<f32>,
    d3: Vec<f32>,
    d4: Vec<f32>,
    kosul_v: Vec<f32>,
    kosul_u: Vec<f32>,
    p1: Vec<usize>,
    p2: Vec<usize>,
}

impl Hadamard {
    /// A fresh block: Walsh factors, unit diagonals, zero bias, zero
    /// conditioning output.
    ///
    /// # Errors
    ///
    /// Whatever [`HadamardSekli::yeni`] refuses.
    pub fn yeni(
        d_model: usize,
        tohum: &mut Tohum,
        yariya_ayrik: bool,
    ) -> Result<Self, HadamardHatasi> {
        let sekil = HadamardSekli::yeni(d_model)?;
        let wa = walsh(sekil.ba);
        let wb = walsh(sekil.bb);
        let faktorler = [(wa.clone(), wb.clone()), (wa.clone(), wb.clone()), (wa, wb)];
        let kosul_v: Vec<f32> = (0..d_model * KOSUL_RANK)
            .map(|_| (tohum.normal() * f64::from(KOSUL_INIT_STD)) as f32)
            .collect();
        Ok(Self {
            sekil,
            faktorler,
            d1: vec![1.0; sekil.n],
            d2: vec![1.0; sekil.n],
            b2: vec![0.0; sekil.n],
            d3: vec![1.0; sekil.n],
            d4: vec![CIKIS_DIAGONAL_INIT; sekil.n],
            kosul_v,
            kosul_u: vec![0.0; KOSUL_RANK * sekil.n],
            p1: karisim(sekil.n, KARISIM_TOHUMLARI.0, yariya_ayrik),
            p2: karisim(sekil.n, KARISIM_TOHUMLARI.1, yariya_ayrik),
        })
    }

    #[must_use]
    pub fn sekil(&self) -> HadamardSekli {
        self.sekil
    }

    /// The parameters actually held, counted by walking the buffers.
    ///
    /// The second opinion to [`HadamardSekli::param_sayisi`]: a formula and a
    /// walk can disagree, and when they do one of them is wrong.
    #[must_use]
    pub fn tutulan_param_sayisi(&self) -> usize {
        self.faktorler
            .iter()
            .map(|(a, b)| a.len() + b.len())
            .sum::<usize>()
            + self.d1.len()
            + self.d2.len()
            + self.b2.len()
            + self.d3.len()
            + self.d4.len()
            + self.kosul_v.len()
            + self.kosul_u.len()
    }

    /// The conditioning vector for one token: `1 + softmax(x W_v) W_u`.
    #[must_use]
    pub fn kosul(&self, x: &[f32]) -> Vec<f32> {
        let mut puan = vec![0.0f32; KOSUL_RANK];
        for (i, deger) in x.iter().enumerate() {
            let satir = &self.kosul_v[i * KOSUL_RANK..(i + 1) * KOSUL_RANK];
            for (p, w) in puan.iter_mut().zip(satir.iter()) {
                *p += deger * w;
            }
        }
        let agirlik = yumusak_azami(&puan);
        let mut c = vec![1.0f32; self.sekil.n];
        for (r, a) in agirlik.iter().enumerate() {
            if *a == 0.0 {
                continue;
            }
            let satir = &self.kosul_u[r * self.sekil.n..(r + 1) * self.sekil.n];
            for (hedef, u) in c.iter_mut().zip(satir.iter()) {
                *hedef += a * u;
            }
        }
        c
    }

    /// The forward pass over a sequence laid out as `jeton * d_model`.
    ///
    /// # Errors
    ///
    /// [`HadamardHatasi::GirdiSekli`] when the input is not a whole number of
    /// tokens. Every token is transformed on its own, which
    /// `jetonlar_bagimsiz` measures - a feed-forward that leaked across
    /// positions would still produce plausible numbers.
    pub fn ileri(&self, x: &[f32]) -> Result<Vec<f32>, HadamardHatasi> {
        let d = self.sekil.d_model;
        if x.len() % d != 0 {
            return Err(HadamardHatasi::GirdiSekli {
                beklenen_kati: d,
                gelen: x.len(),
            });
        }
        let n = self.sekil.n;
        let (ba, bb) = (self.sekil.ba, self.sekil.bb);
        let mut cikis = Vec::with_capacity(x.len());
        for jeton in x.chunks_exact(d) {
            let c = self.kosul(jeton);
            let mut z = vec![0.0f32; n];
            z[..d].copy_from_slice(jeton);

            for (v, w) in z.iter_mut().zip(self.d1.iter()) {
                *v *= w;
            }
            let (a, b) = &self.faktorler[0];
            z = kron_uygula(&z, a, ba, b, bb);
            z = self.p1.iter().map(|i| z[*i]).collect();

            for ((v, w), (ci, bi)) in z
                .iter_mut()
                .zip(self.d2.iter())
                .zip(c.iter().zip(self.b2.iter()))
            {
                *v = silu(*v * *w * *ci + *bi);
            }
            let (a, b) = &self.faktorler[1];
            z = kron_uygula(&z, a, ba, b, bb);
            z = self.p2.iter().map(|i| z[*i]).collect();

            for (v, w) in z.iter_mut().zip(self.d3.iter()) {
                *v *= w;
            }
            let (a, b) = &self.faktorler[2];
            z = kron_uygula(&z, a, ba, b, bb);

            for (v, w) in z.iter_mut().zip(self.d4.iter()) {
                *v *= w;
            }
            cikis.extend_from_slice(&z[..d]);
        }
        Ok(cikis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hadamard(d: usize) -> Hadamard {
        let mut t = Tohum::yeni(7);
        match Hadamard::yeni(d, &mut t, false) {
            Ok(h) => h,
            Err(e) => panic!("kurulmadi: {e}"),
        }
    }

    #[test]
    fn walsh_boyut_bir() {
        assert_eq!(walsh(1), vec![1.0]);
    }

    #[test]
    fn walsh_boyut_iki() {
        let h = walsh(2);
        let s = 1.0 / 2.0f32.sqrt();
        assert_eq!(h, vec![s, s, s, -s]);
    }

    #[test]
    fn walsh_dik() {
        for n in [1usize, 2, 4, 8, 16] {
            let h = walsh(n);
            for i in 0..n {
                for j in 0..n {
                    let ic: f32 = (0..n).map(|k| h[i * n + k] * h[j * n + k]).sum();
                    let beklenen = if i == j { 1.0 } else { 0.0 };
                    assert!((ic - beklenen).abs() < 1e-5, "n={n} i={i} j={j} ic={ic}");
                }
            }
        }
    }

    #[test]
    fn walsh_kendi_tersi() {
        // Symmetric and orthogonal, so H H = I: three stages cannot grow the
        // signal by sqrt(n) each time.
        let n = 8;
        let h = walsh(n);
        for i in 0..n {
            for j in 0..n {
                let hh: f32 = (0..n).map(|k| h[i * n + k] * h[k * n + j]).sum();
                let beklenen = if i == j { 1.0 } else { 0.0 };
                assert!((hh - beklenen).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn walsh_girdileri_isaretli_olcek() {
        let n = 16;
        let olcek = 1.0 / (n as f32).sqrt();
        for v in walsh(n) {
            assert!((v.abs() - olcek).abs() < 1e-6);
        }
    }

    #[test]
    fn blok_bol_kare_gibi() {
        assert_eq!(blok_bol(1024), (32, 32));
        assert_eq!(blok_bol(512), (16, 32));
        assert_eq!(blok_bol(256), (16, 16));
        assert_eq!(blok_bol(2), (1, 2));
    }

    #[test]
    fn blok_bol_carpim_korunur() {
        for k in 0..12 {
            let n = 1usize << k;
            let (a, b) = blok_bol(n);
            assert_eq!(a * b, n, "n={n}");
        }
    }

    #[test]
    fn kron_yogun_esdeger() {
        // The factored application equals a dense multiply by kron(A, B).
        let (ba, bb) = (2usize, 4usize);
        let a = walsh(ba);
        let b = walsh(bb);
        let z: Vec<f32> = (0..ba * bb).map(|i| (i as f32) * 0.25 - 1.0).collect();
        let hizli = kron_uygula(&z, &a, ba, &b, bb);
        let n = ba * bb;
        let mut yogun = vec![0.0f32; n];
        for k in 0..ba {
            for l in 0..bb {
                let mut toplam = 0.0f32;
                for i in 0..ba {
                    for j in 0..bb {
                        toplam += z[i * bb + j] * a[i * ba + k] * b[j * bb + l];
                    }
                }
                yogun[k * bb + l] = toplam;
            }
        }
        for (h, y) in hizli.iter().zip(yogun.iter()) {
            assert!((h - y).abs() < 1e-5, "{h} vs {y}");
        }
    }

    #[test]
    fn kron_birim_faktorler_degistirmez() {
        let (ba, bb) = (2usize, 2usize);
        let birim_a = vec![1.0, 0.0, 0.0, 1.0];
        let birim_b = vec![1.0, 0.0, 0.0, 1.0];
        let z = vec![1.0, -2.0, 3.0, 0.5];
        assert_eq!(kron_uygula(&z, &birim_a, ba, &birim_b, bb), z);
    }

    #[test]
    fn karisim_gercek_permutasyon() {
        for n in [1usize, 2, 8, 64, 1024] {
            let p = karisim(n, 11, false);
            assert_eq!(p.len(), n);
            let mut gorulen = vec![false; n];
            for i in p {
                assert!(!gorulen[i], "n={n}: tekrarlanan indeks");
                gorulen[i] = true;
            }
            assert!(gorulen.into_iter().all(|v| v));
        }
    }

    #[test]
    fn karisim_belirlenimci() {
        assert_eq!(karisim(64, 11, false), karisim(64, 11, false));
    }

    #[test]
    fn karisim_tohumlar_ayri() {
        // Two mixing steps with one permutation would be one mixing step.
        assert_ne!(
            karisim(64, KARISIM_TOHUMLARI.0, false),
            karisim(64, KARISIM_TOHUMLARI.1, false)
        );
    }

    #[test]
    fn karisim_yariya_ayrik_yariyi_gecmez() {
        let n = 64;
        let p = karisim(n, 11, true);
        for (yer, kaynak) in p.iter().enumerate() {
            assert_eq!(yer < n / 2, *kaynak < n / 2, "yer={yer}");
        }
    }

    #[test]
    fn silu_sifirda_sifir() {
        assert!(silu(0.0).abs() < 1e-7);
    }

    #[test]
    fn silu_buyukte_dogrusal() {
        assert!((silu(20.0) - 20.0).abs() < 1e-4);
    }

    #[test]
    fn silu_negatifte_kucuk_ve_isaretli() {
        assert!(silu(-20.0) < 0.0);
        assert!(silu(-20.0).abs() < 1e-6);
    }

    #[test]
    fn yumusak_azami_bire_toplanir() {
        let p = yumusak_azami(&[1.0, 2.0, 3.0]);
        let toplam: f32 = p.iter().sum();
        assert!((toplam - 1.0).abs() < 1e-6);
    }

    #[test]
    fn yumusak_azami_buyuk_sayida_tasmaz() {
        let p = yumusak_azami(&[1.0e30, 1.0e30 + 1.0]);
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sekil_sifir_genisligi_reddeder() {
        assert_eq!(HadamardSekli::yeni(0), Err(HadamardHatasi::SifirGenislik));
    }

    #[test]
    fn sekil_ikinin_kuvvetine_doldurur() {
        let s = match HadamardSekli::yeni(768) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(s.n, 1024);
        assert_eq!((s.ba, s.bb), (32, 32));
    }

    #[test]
    fn param_sayisi_iki_yoldan_ayni() {
        for d in [16usize, 48, 768] {
            let h = hadamard(d);
            assert_eq!(
                h.sekil().param_sayisi(),
                h.tutulan_param_sayisi(),
                "d={d}: formul ile yurume ayristi"
            );
        }
    }

    #[test]
    fn yapisal_maliyet_yogundan_ucuz() {
        let s = match HadamardSekli::yeni(768) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        };
        let (yogun, yapisal) = s.carp_tasarrufu();
        assert_eq!(yogun, 1_048_576);
        assert_eq!(yapisal, 6144);
        assert!(yapisal * 100 < yogun);
    }

    #[test]
    fn kosul_sifirken_birdir() {
        // Bit-identical, not approximately: a fresh conditioning path is a
        // no-op and the test says so with to_bits.
        let h = hadamard(32);
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.0).collect();
        for v in h.kosul(&x) {
            assert_eq!(v.to_bits(), 1.0f32.to_bits());
        }
    }

    #[test]
    fn cikis_genisligi_girdiyle_ayni() {
        let h = hadamard(48);
        let x = vec![0.5f32; 48 * 3];
        match h.ileri(&x) {
            Ok(y) => assert_eq!(y.len(), 48 * 3),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn sifir_girdi_sifir_cikti() {
        // b2 is zero and silu(0) is zero, so a fresh block maps zero to zero
        // exactly; a stray bias would show up here as a nonzero output.
        let h = hadamard(16);
        let y = match h.ileri(&[0.0f32; 16]) {
            Ok(y) => y,
            Err(e) => panic!("{e}"),
        };
        for v in y {
            assert_eq!(v.to_bits(), 0.0f32.to_bits());
        }
    }

    #[test]
    fn ileri_belirlenimci() {
        let h = hadamard(32);
        let x: Vec<f32> = (0..64).map(|i| ((i % 7) as f32) * 0.3 - 1.0).collect();
        let (a, b) = match (h.ileri(&x), h.ileri(&x)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => panic!("ileri reddetti"),
        };
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn jetonlar_bagimsiz() {
        let h = hadamard(16);
        let bir: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let iki: Vec<f32> = (0..16).map(|i| 1.0 - (i as f32) * 0.05).collect();
        let mut birlikte = bir.clone();
        birlikte.extend_from_slice(&iki);
        let (y_birlikte, y_bir, y_iki) = match (h.ileri(&birlikte), h.ileri(&bir), h.ileri(&iki)) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => panic!("ileri reddetti"),
        };
        for (yer, v) in y_bir.iter().enumerate() {
            assert_eq!(y_birlikte[yer].to_bits(), v.to_bits());
        }
        for (yer, v) in y_iki.iter().enumerate() {
            assert_eq!(y_birlikte[16 + yer].to_bits(), v.to_bits());
        }
    }

    #[test]
    fn eksik_jeton_reddedilir() {
        let h = hadamard(16);
        assert_eq!(
            h.ileri(&[0.0f32; 17]),
            Err(HadamardHatasi::GirdiSekli {
                beklenen_kati: 16,
                gelen: 17
            })
        );
    }

    #[test]
    fn cikis_diagonali_kucuk_baslar() {
        // The block starts quiet: with unit diagonals elsewhere the output
        // scale is the output diagonal, and that is 0.02 rather than 1.
        let h = hadamard(16);
        let x: Vec<f32> = (0..16).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let y = match h.ileri(&x) {
            Ok(y) => y,
            Err(e) => panic!("{e}"),
        };
        let enb = y.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(enb < CIKIS_DIAGONAL_INIT, "cikis olcegi {enb}");
    }

    #[test]
    fn hata_metinleri_ayirt_edilir() {
        assert_ne!(
            HadamardHatasi::SifirGenislik.to_string(),
            HadamardHatasi::GirdiSekli {
                beklenen_kati: 2,
                gelen: 3
            }
            .to_string()
        );
    }
}
