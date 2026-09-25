//! The checkpoint's own description of itself.
//!
//! # Why the configuration is read rather than assumed
//!
//! Every number a forward pass needs is in the file next to the weights: how
//! many layers, how wide, which layers attend globally and which attend inside
//! a window, the rope base, the vocabulary size. Hard-coding them would make
//! the port correct for exactly one checkpoint and silently wrong for the next
//! one, and the failure would show up as slightly different numbers rather than
//! as an error. A field the port does not understand is kept and reported, not
//! dropped: an unknown key is a hint that the checkpoint expects behaviour this
//! build does not have.
//!
//! # The decision head's configuration
//!
//! The head is trained by reinforcement, and its file carries the economics
//! rather than only the shapes: what an escalation costs, what a wrong action
//! costs. Those numbers are what a policy is *for*; a port that read the shapes
//! and ignored the costs would run the head and have no way to say whether its
//! output was good.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

/// One attention layer's kind, as the configuration names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KatmanTuru {
    /// Attend over the whole sequence.
    FullAttention,
    /// Attend inside a window of `local_attention` positions.
    SlidingAttention,
}

impl KatmanTuru {
    /// The label a report prints.
    #[must_use]
    pub fn ad(self) -> &'static str {
        match self {
            Self::FullAttention => "tam",
            Self::SlidingAttention => "kayan",
        }
    }
}

/// The encoder's structural configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct KodlayiciYapisi {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden width.
    pub hidden_size: usize,
    /// Attention heads.
    pub num_attention_heads: usize,
    /// Feed-forward width (the gated projection's *total* width).
    pub intermediate_size: usize,
    /// Layers.
    pub num_hidden_layers: usize,
    /// The kind of each layer, in order.
    pub layer_types: Vec<KatmanTuru>,
    /// Rope base; the configuration stores it per attention kind.
    #[serde(default)]
    pub rope_parameters: BTreeMap<String, RopeAyari>,
    /// Window length for sliding layers.
    #[serde(default = "varsayilan_pencere")]
    pub local_attention: usize,
    /// Which positions the window covers. Not a checkpoint field: the file
    /// states the width and not the shape, so the shape is carried here,
    /// labelled, and settled by the cross-check rather than by a guess.
    #[serde(skip)]
    pub pencere_kurali: crate::blok::PencereKurali,
    /// Epsilon used by every normalisation.
    #[serde(default = "varsayilan_epsilon")]
    pub layer_norm_eps: f32,
    /// Maximum sequence length.
    #[serde(default)]
    pub max_position_embeddings: usize,
    /// Activation name; only `gelu` is implemented, and anything else is
    /// refused rather than approximated.
    #[serde(default = "varsayilan_aktivasyon")]
    pub hidden_activation: String,
    /// Token ids the tokenizer needs.
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    /// See [`Self::pad_token_id`].
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// See [`Self::pad_token_id`].
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// See [`Self::pad_token_id`].
    #[serde(default)]
    pub mask_token_id: Option<u32>,
    /// Whether the embedding matrix is reused as the output projection.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Pooling the classic head used, kept for the record.
    #[serde(default)]
    pub classifier_pooling: Option<String>,
    /// Keys this build does not read. Kept so that an unexpected field can be
    /// reported instead of vanishing.
    #[serde(flatten)]
    pub diger: BTreeMap<String, serde_json::Value>,
}

fn varsayilan_pencere() -> usize {
    128
}

fn varsayilan_epsilon() -> f32 {
    1.0e-5
}

fn varsayilan_aktivasyon() -> String {
    "gelu".to_string()
}

/// One rope entry.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeAyari {
    /// The base, 10 000 unless the model was trained otherwise.
    #[serde(default = "varsayilan_theta")]
    pub rope_theta: f32,
    /// The kind; only the default is implemented.
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn varsayilan_theta() -> f32 {
    10_000.0
}

impl KodlayiciYapisi {
    /// The rope base for a layer kind.
    #[must_use]
    pub fn theta(&self, tur: KatmanTuru) -> f32 {
        let anahtar = match tur {
            KatmanTuru::FullAttention => "full_attention",
            KatmanTuru::SlidingAttention => "sliding_attention",
        };
        self.rope_parameters
            .get(anahtar)
            .map_or(10_000.0, |r| r.rope_theta)
    }

    /// How wide one head is.
    #[must_use]
    pub fn kafa_genisligi(&self) -> usize {
        self.hidden_size.checked_div(self.num_attention_heads).unwrap_or(0)
    }

    /// Checks the parts of the configuration the forward pass depends on.
    ///
    /// Each check is a shape the arithmetic would otherwise discover later, as
    /// a wrong number rather than as an error.
    ///
    /// # Errors
    /// A sentence naming the field that does not hold.
    pub fn dogrula(&self) -> Result<(), String> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            return Err("hidden_size ve num_attention_heads sifir olamaz".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "hidden_size {} num_attention_heads {} ile bolunmuyor",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types {} katman yaziyor, num_hidden_layers {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.hidden_activation != "gelu" {
            return Err(format!(
                "aktivasyon `{}` bu yapida yok; gelu beklenir",
                self.hidden_activation
            ));
        }
        if self.local_attention == 0 {
            return Err("local_attention sifir olamaz".to_string());
        }
        // A rotary table with an odd head width has no pair to rotate.
        if !self.kafa_genisligi().is_multiple_of(2) {
            return Err(format!(
                "kafa genisligi {} cift degil",
                self.kafa_genisligi()
            ));
        }
        Ok(())
    }

    /// Reads a configuration file.
    ///
    /// # Errors
    /// The path or the parser, named.
    pub fn oku(yol: &Path) -> Result<Self, String> {
        let metin = std::fs::read_to_string(yol).map_err(|h| format!("{}: {h}", yol.display()))?;
        let yapi: Self =
            serde_json::from_str(&metin).map_err(|h| format!("{}: {h}", yol.display()))?;
        yapi.dogrula()?;
        Ok(yapi)
    }
}

/// The decision head's economics, from its own file.
#[derive(Debug, Clone, Deserialize)]
pub struct KafaYapisi {
    /// Turns of history the head reads.
    #[serde(default)]
    pub head_layers: usize,
    /// Longest input the encoder sees.
    #[serde(default)]
    pub max_len: usize,
    /// Longest input the head sees.
    #[serde(default)]
    pub head_max_len: usize,
    /// How many prefixes may be under consideration at once.
    #[serde(default)]
    pub max_prefixes: usize,
    /// What an escalation costs.
    #[serde(default)]
    pub act_costs: BTreeMap<String, f64>,
    /// What a wrong action costs.
    #[serde(default)]
    pub cost_wrong_act: f64,
    /// Temperature per position, as trained.
    #[serde(default)]
    pub temperature: Vec<f64>,
    /// Keys this build does not read; see [`KodlayiciYapisi::diger`].
    #[serde(flatten)]
    pub diger: BTreeMap<String, serde_json::Value>,
}

impl KafaYapisi {
    /// The cost of escalating, or `None` when the file did not state one.
    #[must_use]
    pub fn escalate_maliyeti(&self) -> Option<f64> {
        self.act_costs.get("escalate").copied()
    }

    /// The break-even accuracy an action must beat for acting to be worth it.
    ///
    /// This is the number the economics imply rather than a threshold anyone
    /// chose: acting is worth it when
    /// `p * (0) + (1 - p) * wrong > escalate`, that is when
    /// `1 - p < escalate / wrong`, so the action must be right more than
    /// `1 - escalate / wrong` of the time. Reporting it is the difference
    /// between carrying the costs and using them.
    #[must_use]
    pub fn esik_dogruluk(&self) -> Option<f64> {
        let escalate = self.escalate_maliyeti()?;
        if self.cost_wrong_act <= 0.0 {
            return None;
        }
        Some(1.0 - escalate / self.cost_wrong_act)
    }

    /// Reads a head configuration file.
    ///
    /// # Errors
    /// The path or the parser, named.
    pub fn oku(yol: &Path) -> Result<Self, String> {
        let metin = std::fs::read_to_string(yol).map_err(|h| format!("{}: {h}", yol.display()))?;
        serde_json::from_str(&metin).map_err(|h| format!("{}: {h}", yol.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn gecici(ad: &str, icerik: &str) -> PathBuf {
        let yol = std::env::temp_dir().join(format!("lubot-kodlayici-yapi-{ad}.json"));
        std::fs::write(&yol, icerik).expect("yazilmali");
        yol
    }

    const ORNEK: &str = r#"{
        "vocab_size": 256000,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 1152,
        "num_hidden_layers": 3,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "local_attention": 128,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 8192,
        "hidden_activation": "gelu",
        "tie_word_embeddings": true,
        "rope_parameters": {
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"}
        },
        "classifier_pooling": "mean",
        "model_type": "modernbert",
        "position_embedding_type": "sans_pos"
    }"#;

    #[test]
    fn the_shape_is_read_from_the_file_and_not_assumed() {
        let yol = gecici("ornek", ORNEK);
        let yapi = KodlayiciYapisi::oku(&yol).expect("okunmali");
        assert_eq!(yapi.hidden_size, 768);
        assert_eq!(yapi.kafa_genisligi(), 64);
        assert_eq!(yapi.theta(KatmanTuru::FullAttention), 160_000.0);
        assert_eq!(yapi.layer_types[1], KatmanTuru::SlidingAttention);
        assert!(yapi.tie_word_embeddings);
        // Keys this build does not read are kept, not dropped.
        assert!(yapi.diger.contains_key("model_type"));
        assert!(yapi.diger.contains_key("position_embedding_type"));
        let _ = std::fs::remove_file(&yol);
    }

    #[test]
    fn a_layer_count_that_disagrees_with_the_types_is_refused() {
        let bozuk = ORNEK.replace("\"num_hidden_layers\": 3", "\"num_hidden_layers\": 4");
        let yol = gecici("katman", &bozuk);
        let hata = KodlayiciYapisi::oku(&yol).unwrap_err();
        assert!(hata.contains("layer_types"), "{hata}");
        let _ = std::fs::remove_file(&yol);
    }

    #[test]
    fn an_activation_this_build_does_not_have_is_refused_rather_than_approximated() {
        let bozuk = ORNEK.replace("\"gelu\"", "\"swiglu\"");
        let yol = gecici("aktivasyon", &bozuk);
        let hata = KodlayiciYapisi::oku(&yol).unwrap_err();
        assert!(hata.contains("swiglu"), "{hata}");
        let _ = std::fs::remove_file(&yol);
    }

    #[test]
    fn a_head_width_that_cannot_be_rotated_is_refused() {
        let bozuk = ORNEK.replace("\"num_attention_heads\": 12", "\"num_attention_heads\": 5");
        // 768 / 5 is not an integer, so the division check fires first.
        let yol = gecici("kafa", &bozuk);
        assert!(KodlayiciYapisi::oku(&yol).is_err());
        // 18 / 2 = 9: it divides, so the division check passes and the odd head
        // width is what has to be caught - a rotary table needs pairs.
        let bozuk2 = ORNEK
            .replace("\"hidden_size\": 768", "\"hidden_size\": 18")
            .replace("\"num_attention_heads\": 12", "\"num_attention_heads\": 2");
        let yol2 = gecici("kafa2", &bozuk2);
        let hata2 = KodlayiciYapisi::oku(&yol2).unwrap_err();
        assert!(hata2.contains("cift degil"), "{hata2}");
        let _ = std::fs::remove_file(&yol);
        let _ = std::fs::remove_file(&yol2);
    }

    #[test]
    fn the_break_even_accuracy_follows_from_the_stated_costs() {
        let yol = gecici(
            "kafa",
            r#"{"head_layers": 2, "max_len": 1024, "head_max_len": 256,
                "max_prefixes": 6, "act_costs": {"escalate": 0.5},
                "cost_wrong_act": 3.0, "temperature": [1.0, 1.0, 1.0]}"#,
        );
        let kafa = KafaYapisi::oku(&yol).expect("okunmali");
        assert_eq!(kafa.escalate_maliyeti(), Some(0.5));
        // 1 - 0.5/3 = 0.8333...: above this, acting beats escalating.
        let esik = kafa.esik_dogruluk().expect("esik hesaplanmali");
        assert!((esik - 0.833_333_3).abs() < 1e-6, "{esik}");
        let _ = std::fs::remove_file(&yol);
    }

    #[test]
    fn a_head_without_costs_reports_no_break_even_rather_than_a_default_one() {
        let yol = gecici("kafasiz", r#"{"head_layers": 2}"#);
        let kafa = KafaYapisi::oku(&yol).expect("okunmali");
        assert!(kafa.escalate_maliyeti().is_none());
        assert!(kafa.esik_dogruluk().is_none());
        let _ = std::fs::remove_file(&yol);
    }
}
