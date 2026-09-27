//! The attention schedule: which layers see everything, which see a window.
//!
//! # The pattern
//!
//! Every layer attending to every position costs `O(n^2)` per layer, and most
//! of that cost buys nothing: a token's neighbours carry most of what it needs.
//! The pattern this module encodes keeps a sliding window on most layers and
//! opens the whole sequence on a periodic few, so information still crosses the
//! sequence - it just crosses it every `periyot` layers instead of every layer.
//!
//! # Why the schedule is a value
//!
//! The rule "one layer in three is global" is one integer. Written as an `if`
//! inside the forward pass it is one integer nobody can read off without
//! running the model, and a model whose attention pattern can only be recovered
//! by instrumenting it is a model whose attention pattern nobody checks.
//! [`Plan`] is built once, can be printed, and is what the layer loop consults.
//!
//! # Bidirectional, not causal
//!
//! This backbone reads; it does not continue text. The window is therefore
//! symmetric - a token sees `yaricap` positions on each side - and there is no
//! causal mask. A causal variant is a different model family and is not smuggled
//! in here as a flag.

/// What one layer is allowed to see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kapsam {
    /// The whole sequence.
    Genel,
    /// A symmetric window of this radius on each side.
    Yerel { yaricap: usize },
}

impl Kapsam {
    /// Whether a query at `sorgu` may attend to a key at `anahtar`.
    ///
    /// The rule lives here, once. [`Plan::gorulebilir`] delegates to it, and so
    /// does the forward pass, so there is no second copy to drift.
    #[must_use]
    pub fn gorulebilir(self, sorgu: usize, anahtar: usize) -> bool {
        match self {
            Self::Genel => true,
            Self::Yerel { yaricap } => sorgu.abs_diff(anahtar) <= yaricap,
        }
    }
}

/// Why a schedule was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PencereHatasi {
    /// A model with no layers has no schedule.
    SifirKatman,
    /// A period of zero would divide by zero; a period of one makes every layer
    /// global, which is the pattern this module exists to avoid being an
    /// accident.
    GecersizPeriyot { periyot: usize },
    /// A window of radius zero lets a token see only itself, which is not
    /// attention.
    SifirYaricap,
    /// The layer index is past the end of the schedule.
    KatmanYok { katman: usize, n_katman: usize },
}

impl std::fmt::Display for PencereHatasi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SifirKatman => write!(f, "a model with no layers has no attention schedule"),
            Self::GecersizPeriyot { periyot } => write!(
                f,
                "a global period of {periyot} is not a schedule; use two or more"
            ),
            Self::SifirYaricap => write!(
                f,
                "a window of radius zero lets a token see only itself, which is not attention"
            ),
            Self::KatmanYok { katman, n_katman } => write!(
                f,
                "layer {katman} is past the end of a {n_katman}-layer schedule"
            ),
        }
    }
}

/// The per-layer schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    katmanlar: Vec<Kapsam>,
    periyot: usize,
    yaricap: usize,
}

impl Plan {
    /// One layer in `periyot` is global, counting from layer zero; the rest
    /// carry a symmetric window of `yaricap`.
    ///
    /// Layer zero is global by construction. The first layer is where the
    /// sequence is still raw embeddings, and starting the stack with a window
    /// means the first mixing a token ever sees is already truncated.
    ///
    /// # Errors
    ///
    /// [`PencereHatasi::SifirKatman`], [`PencereHatasi::GecersizPeriyot`] or
    /// [`PencereHatasi::SifirYaricap`].
    pub fn periyodik(
        n_katman: usize,
        periyot: usize,
        yaricap: usize,
    ) -> Result<Self, PencereHatasi> {
        if n_katman == 0 {
            return Err(PencereHatasi::SifirKatman);
        }
        if periyot < 2 {
            return Err(PencereHatasi::GecersizPeriyot { periyot });
        }
        if yaricap == 0 {
            return Err(PencereHatasi::SifirYaricap);
        }
        let katmanlar = (0..n_katman)
            .map(|i| {
                if i.is_multiple_of(periyot) {
                    Kapsam::Genel
                } else {
                    Kapsam::Yerel { yaricap }
                }
            })
            .collect();
        Ok(Self {
            katmanlar,
            periyot,
            yaricap,
        })
    }

    #[must_use]
    pub fn n_katman(&self) -> usize {
        self.katmanlar.len()
    }

    #[must_use]
    pub fn periyot(&self) -> usize {
        self.periyot
    }

    #[must_use]
    pub fn yaricap(&self) -> usize {
        self.yaricap
    }

    /// What layer `katman` may see.
    ///
    /// # Errors
    ///
    /// [`PencereHatasi::KatmanYok`].
    pub fn kapsam(&self, katman: usize) -> Result<Kapsam, PencereHatasi> {
        self.katmanlar
            .get(katman)
            .copied()
            .ok_or(PencereHatasi::KatmanYok {
                katman,
                n_katman: self.katmanlar.len(),
            })
    }

    /// How many layers see the whole sequence.
    #[must_use]
    pub fn genel_sayisi(&self) -> usize {
        self.katmanlar
            .iter()
            .filter(|k| matches!(k, Kapsam::Genel))
            .count()
    }

    /// Whether a query at `sorgu` may attend to a key at `anahtar` in this
    /// layer.
    ///
    /// # Errors
    ///
    /// [`PencereHatasi::KatmanYok`].
    pub fn gorulebilir(
        &self,
        katman: usize,
        sorgu: usize,
        anahtar: usize,
    ) -> Result<bool, PencereHatasi> {
        Ok(self.kapsam(katman)?.gorulebilir(sorgu, anahtar))
    }

    /// The schedule as a short string, for a signature or a report.
    #[must_use]
    pub fn ozet(&self) -> String {
        self.katmanlar
            .iter()
            .map(|k| match k {
                Kapsam::Genel => 'G',
                Kapsam::Yerel { .. } => 'y',
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sifir_katman_reddedilir() {
        assert_eq!(Plan::periyodik(0, 3, 64), Err(PencereHatasi::SifirKatman));
    }

    #[test]
    fn gecersiz_periyot_reddedilir() {
        assert_eq!(
            Plan::periyodik(6, 1, 64),
            Err(PencereHatasi::GecersizPeriyot { periyot: 1 })
        );
        assert_eq!(
            Plan::periyodik(6, 0, 64),
            Err(PencereHatasi::GecersizPeriyot { periyot: 0 })
        );
    }

    #[test]
    fn sifir_yaricap_reddedilir() {
        assert_eq!(Plan::periyodik(6, 3, 0), Err(PencereHatasi::SifirYaricap));
    }

    #[test]
    fn ilk_katman_geneldir() {
        let plan = Plan::periyodik(9, 3, 32).unwrap();
        assert_eq!(plan.kapsam(0), Ok(Kapsam::Genel));
    }

    #[test]
    fn her_ucte_bir_genel() {
        let plan = Plan::periyodik(9, 3, 32).unwrap();
        assert_eq!(plan.ozet(), "GyyGyyGyy");
        assert_eq!(plan.genel_sayisi(), 3);
    }

    #[test]
    fn genel_sayisi_tavan_bolme() {
        // Ten layers with period three: 0, 3, 6, 9 - four, not three.
        let plan = Plan::periyodik(10, 3, 32).unwrap();
        assert_eq!(plan.genel_sayisi(), 4);
        assert_eq!(plan.ozet(), "GyyGyyGyyG");
    }

    #[test]
    fn katman_disina_sorulmaz() {
        let plan = Plan::periyodik(4, 2, 8).unwrap();
        assert_eq!(
            plan.kapsam(4),
            Err(PencereHatasi::KatmanYok {
                katman: 4,
                n_katman: 4
            })
        );
        assert_eq!(
            plan.gorulebilir(9, 0, 0),
            Err(PencereHatasi::KatmanYok {
                katman: 9,
                n_katman: 4
            })
        );
    }

    #[test]
    fn genel_katman_her_seyi_gorur() {
        let plan = Plan::periyodik(3, 3, 1).unwrap();
        for anahtar in 0..50 {
            assert_eq!(plan.gorulebilir(0, 0, anahtar), Ok(true));
        }
    }

    #[test]
    fn yerel_katman_pencereyle_sinirli() {
        let plan = Plan::periyodik(3, 3, 2).unwrap();
        assert_eq!(plan.gorulebilir(1, 10, 8), Ok(true));
        assert_eq!(plan.gorulebilir(1, 10, 12), Ok(true));
        assert_eq!(plan.gorulebilir(1, 10, 7), Ok(false));
        assert_eq!(plan.gorulebilir(1, 10, 13), Ok(false));
    }

    #[test]
    fn pencere_simetrik() {
        let plan = Plan::periyodik(3, 3, 4).unwrap();
        for a in 0..20usize {
            for b in 0..20usize {
                assert_eq!(
                    plan.gorulebilir(1, a, b),
                    plan.gorulebilir(1, b, a),
                    "the window is not symmetric at ({a}, {b})"
                );
            }
        }
    }

    #[test]
    fn her_jeton_kendini_gorur() {
        // If this ever fails, softmax has a row of nothing but -inf and the
        // forward pass is undefined rather than merely wrong.
        let plan = Plan::periyodik(7, 3, 1).unwrap();
        for katman in 0..plan.n_katman() {
            for pos in 0..30usize {
                assert_eq!(plan.gorulebilir(katman, pos, pos), Ok(true));
            }
        }
    }

    #[test]
    fn alanlar_geri_okunur() {
        let plan = Plan::periyodik(5, 4, 17).unwrap();
        assert_eq!(plan.n_katman(), 5);
        assert_eq!(plan.periyot(), 4);
        assert_eq!(plan.yaricap(), 17);
    }

    #[test]
    fn deterministik() {
        assert_eq!(
            Plan::periyodik(11, 3, 64).unwrap(),
            Plan::periyodik(11, 3, 64).unwrap()
        );
    }

    #[test]
    fn hata_metinleri_bos_degil() {
        assert!(!PencereHatasi::SifirKatman.to_string().is_empty());
        assert!(PencereHatasi::GecersizPeriyot { periyot: 1 }
            .to_string()
            .contains('1'));
        assert!(!PencereHatasi::SifirYaricap.to_string().is_empty());
        assert!(PencereHatasi::KatmanYok {
            katman: 4,
            n_katman: 4
        }
        .to_string()
        .contains('4'));
    }
}
