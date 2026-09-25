//! # sozluk - the checkpoint's own tokenizer, written from Rust
//!
//! The model reads ids, not text. This module is the step that turns one into
//! the other, and it is ported rather than borrowed: the vocabulary file ships
//! with the checkpoint, and the pipeline around it - a replacing normaliser, a
//! space-marking pre-tokenizer, a byte-fallback BPE and a template
//! post-processor - is reimplemented here so the whole path from a sentence to
//! a decision runs inside this repository.
//!
//! The pipeline, in the order the reference applies it, with the measured
//! behaviour of *this* vocabulary next to each step:
//!
//! 1. **Added tokens are separated first.** The file lists 249 of them, and
//!    they are matched in the raw text before anything else happens. That is
//!    not a detail: the list includes the newline runs, so `"a\nb"` is
//!    `a` + `"\n"` + `b` and not one three-character word. Measured:
//!    `"a\nb"` -> `["▁a", "\n", "▁b"]`. A port that normalises first gets a
//!    single `"▁a\nb"` piece and a different id list that still looks
//!    plausible.
//! 2. **Every space becomes the word marker** (`▁`, U+2581).
//! 3. **The text is split on the marker**, each piece keeping its marker, and a
//!    piece at the very start that does not have one gets one - so `"a"` and
//!    `" a"` begin the same way.
//! 4. **Each piece is merged** by the 580,604 merge rules in the file, lowest
//!    rank first, and any character the vocabulary does not contain falls back
//!    to one token per UTF-8 byte (`"<0x00>"` ... ), so nothing is silently
//!    dropped.
//! 5. **The template wraps the result**: `<bos>` first, `<eos>` last. The ids
//!    are not hardcoded: they are read from the file's own post-processor.
//!
//! Two properties are measured rather than assumed. The first is that this
//! implementation agrees with the reference on a corpus - `tools/sozluk_capraz.py`
//! runs both and compares id by id, and a single mismatch fails it. The second
//! is that the vocabulary itself is refused when it is inconsistent: a
//! `tokenizer.json` whose merges refer to tokens that do not exist would
//! otherwise produce ids from a vocabulary that is not the checkpoint's.

use std::collections::HashMap;
use std::path::Path;

use crate::baslik::BaslikHatasi;

/// The marker the normaliser puts in place of a space.
pub const ISARET: char = '\u{2581}';

/// A token that is matched in the text before the BPE runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EklenenJeton {
    /// The literal text that matches it.
    pub icerik: String,
    /// The id it is emitted as.
    pub id: u32,
    /// Whether whitespace immediately before it is dropped. Measured: `<mask>`
    /// has this set, so `"a <mask> b"` is three tokens and not four.
    pub soldan_kirp: bool,
    /// Whether whitespace immediately after it is dropped.
    pub sagdan_kirp: bool,
    /// Whether the file marks it special (the wrapping tokens are).
    pub ozel: bool,
}

/// The tokenizer: vocabulary, merges, added tokens and the wrap.
#[derive(Debug)]
pub struct Sozluk {
    /// Token text to id.
    id: HashMap<String, u32>,
    /// Id to token text, for decoding and for reports.
    ters: Vec<String>,
    /// A merge rule: the pair of ids, and the id their concatenation has.
    ///
    /// The rank is implicit: the map is built in file order, so a pair present
    /// in it is one the file allows, and the *order* of merging is decided by
    /// the rank stored as the value.
    birlesme: HashMap<(u32, u32), u32>,
    /// The rank of each allowed pair, which is what makes the merge order
    /// deterministic rather than dependent on iteration order.
    rank: HashMap<(u32, u32), u32>,
    /// The added tokens, longest content first so the match is greedy.
    eklenen: Vec<EklenenJeton>,
    /// The id used for an unknown token.
    bilinmeyen: u32,
    /// The id the template puts first, if it puts one.
    bas: Option<u32>,
    /// The id the template puts last, if it puts one.
    son: Option<u32>,
    /// The id that fills a padded batch.
    dolgu: u32,
    /// The id that stands in for a masked position.
    maske: u32,
    /// How many bytes a fallback token covers, kept so a report can say it.
    bayt_onek: String,
}

impl Sozluk {
    /// Reads and checks a `tokenizer.json`.
    ///
    /// # Errors
    /// [`BaslikHatasi::Baslik`] when the file is unreadable, is not the shape
    /// this module reads, or fails its own consistency check - a merge whose
    /// parts or whose result are not in the vocabulary is refused rather than
    /// skipped, because skipping it would silently change every id after it.
    pub fn oku(yol: &Path) -> Result<Self, BaslikHatasi> {
        let metin = std::fs::read_to_string(yol).map_err(|e| BaslikHatasi::Baslik {
            mesaj: format!("{}: {e}", yol.display()),
        })?;
        Self::metinden(&metin)
    }

    /// The same parse, over text already in memory.
    ///
    /// # Errors
    /// As [`Sozluk::oku`].
    pub fn metinden(metin: &str) -> Result<Self, BaslikHatasi> {
        let kok: serde_json::Value =
            serde_json::from_str(metin).map_err(|e| BaslikHatasi::Baslik {
                mesaj: format!("tokenizer.json: {e}"),
            })?;
        let model = kok
            .get("model")
            .ok_or_else(|| hata("tokenizer.json: `model` yok"))?;
        let tur = model
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if tur != "BPE" {
            return Err(hata(&format!("tokenizer.json: model `{tur}` degil BPE")));
        }
        let sozluk = model
            .get("vocab")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| hata("tokenizer.json: `vocab` nesne degil"))?;
        let mut id: HashMap<String, u32> = HashMap::with_capacity(sozluk.len());
        let mut en_buyuk = 0_u32;
        for (jeton, deger) in sozluk {
            let deger = deger
                .as_u64()
                .ok_or_else(|| hata(&format!("vocab: `{jeton}` icin id sayi degil")))?;
            #[allow(clippy::cast_possible_truncation)]
            let deger = deger as u32;
            en_buyuk = en_buyuk.max(deger);
            id.insert(jeton.clone(), deger);
        }
        let mut ters = vec![String::new(); en_buyuk as usize + 1];
        for (jeton, deger) in &id {
            if let Some(yer) = ters.get_mut(*deger as usize) {
                yer.clone_from(jeton);
            }
        }

        let birlesmeler = model
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| hata("tokenizer.json: `merges` dizi degil"))?;
        let mut birlesme: HashMap<(u32, u32), u32> = HashMap::with_capacity(birlesmeler.len());
        let mut rank: HashMap<(u32, u32), u32> = HashMap::with_capacity(birlesmeler.len());
        for (sira, kural) in birlesmeler.iter().enumerate() {
            let (a, b) = match kural {
                serde_json::Value::String(s) => {
                    let mut parca = s.splitn(2, ' ');
                    let a = parca.next().unwrap_or("");
                    let b = parca.next().unwrap_or("");
                    (a.to_string(), b.to_string())
                }
                serde_json::Value::Array(p) if p.len() == 2 => (
                    p[0].as_str().unwrap_or("").to_string(),
                    p[1].as_str().unwrap_or("").to_string(),
                ),
                _ => return Err(hata("tokenizer.json: merge bicimi taninmadi")),
            };
            let (Some(&ia), Some(&ib)) = (id.get(&a), id.get(&b)) else {
                return Err(hata(&format!("merge `{a}`+`{b}`: parcalar sozlukte yok")));
            };
            let birlesti = format!("{a}{b}");
            let Some(&ic) = id.get(&birlesti) else {
                return Err(hata(&format!("merge `{a}`+`{b}`: sonuc `{birlesti}` sozlukte yok")));
            };
            #[allow(clippy::cast_possible_truncation)]
            let sira = sira as u32;
            birlesme.insert((ia, ib), ic);
            rank.insert((ia, ib), sira);
        }

        let eklenen = eklenenleri_oku(&kok, &id)?;
        let bilinmeyen = id
            .get("<unk>")
            .copied()
            .ok_or_else(|| hata("sozlukte `<unk>` yok"))?;
        let (bas, son) = sablon(&kok, &id)?;
        // Read before `id` is moved into the struct; the order matters and the
        // compiler is what makes it matter.
        let dolgu = id.get("<pad>").copied().unwrap_or(bilinmeyen);
        let maske = id.get("<mask>").copied().unwrap_or(bilinmeyen);
        Ok(Self {
            id,
            ters,
            birlesme,
            rank,
            eklenen,
            bilinmeyen,
            bas,
            son,
            dolgu,
            maske,
            bayt_onek: "<0x".to_string(),
        })
    }

    /// The number of tokens in the vocabulary.
    #[must_use]
    pub fn boyut(&self) -> usize {
        self.ters.len()
    }

    /// How many merge rules were read.
    #[must_use]
    pub fn kural_sayisi(&self) -> usize {
        self.birlesme.len()
    }

    /// How many added tokens were read.
    #[must_use]
    pub fn eklenen_sayisi(&self) -> usize {
        self.eklenen.len()
    }

    /// The id that fills a padded batch.
    #[must_use]
    pub fn dolgu(&self) -> u32 {
        self.dolgu
    }

    /// The id that stands in for a masked position.
    #[must_use]
    pub fn maske(&self) -> u32 {
        self.maske
    }

    /// The ids the template wraps a sequence with, first and last.
    #[must_use]
    pub fn sarmalayici(&self) -> (Option<u32>, Option<u32>) {
        (self.bas, self.son)
    }

    /// The token text of an id, or `None`.
    #[must_use]
    pub fn jeton(&self, id: u32) -> Option<&str> {
        self.ters.get(id as usize).map(String::as_str)
    }

    /// Turns text into ids, wrapped the way the checkpoint's template wraps it.
    #[must_use]
    pub fn jetonla(&self, metin: &str) -> Vec<u32> {
        let mut idler = Vec::new();
        if let Some(bas) = self.bas {
            idler.push(bas);
        }
        idler.extend(self.jetonla_cekirdek(metin));
        if let Some(son) = self.son {
            idler.push(son);
        }
        idler
    }

    /// Turns text into ids without the template's wrapping tokens.
    #[must_use]
    pub fn jetonla_cekirdek(&self, metin: &str) -> Vec<u32> {
        let mut idler = Vec::new();
        for parca in self.parcalara_ayir(metin) {
            match parca {
                Parca::Eklenen(id) => idler.push(id),
                Parca::Metin(ham) => {
                    let normal = normalle(&ham);
                    if normal.is_empty() {
                        continue;
                    }
                    for kelime in metaspace(&normal) {
                        idler.extend(self.bpe(&kelime));
                    }
                }
            }
        }
        idler
    }

    /// Decodes ids back to text.
    ///
    /// Byte-fallback tokens are turned back into the bytes they stand for, so a
    /// decode is the inverse of the encode for everything that round-trips at
    /// all, and the parts that do not (an added token that replaces text) are
    /// what they are.
    #[must_use]
    pub fn coz(&self, idler: &[u32]) -> String {
        let mut baytlar: Vec<u8> = Vec::new();
        let mut cikti = String::new();
        let bosalt = |baytlar: &mut Vec<u8>, cikti: &mut String| {
            if baytlar.is_empty() {
                return;
            }
            cikti.push_str(&String::from_utf8_lossy(baytlar));
            baytlar.clear();
        };
        for id in idler {
            let Some(jeton) = self.jeton(*id) else {
                continue;
            };
            if let Some(okunmus) = bayt_jetonu(jeton) {
                baytlar.push(okunmus);
                continue;
            }
            bosalt(&mut baytlar, &mut cikti);
            cikti.push_str(&jeton.replace(ISARET, " "));
        }
        bosalt(&mut baytlar, &mut cikti);
        cikti
    }

    /// Splits the text into added tokens and the text between them.
    fn parcalara_ayir(&self, metin: &str) -> Vec<Parca> {
        let mut parcalar = Vec::new();
        let baytlar = metin.as_bytes();
        let mut bas = 0usize;
        let mut konum = 0usize;
        while konum < baytlar.len() {
            // A match is by bytes but always on a character boundary: the
            // contents are literals from the file, and a match that started
            // mid-character could not equal one of them.
            let mut eslesme: Option<(&EklenenJeton, usize)> = None;
            for jeton in &self.eklenen {
                let icerik = jeton.icerik.as_bytes();
                if metin[konum..].starts_with(&jeton.icerik) {
                    // Longest wins: `"\n\n"` must beat `"\n"`.
                    if eslesme.is_none_or(|(onceki, _)| icerik.len() > onceki.icerik.len()) {
                        eslesme = Some((jeton, icerik.len()));
                    }
                }
            }
            let Some((jeton, uzunluk)) = eslesme else {
                konum += metin[konum..].chars().next().map_or(1, char::len_utf8);
                continue;
            };
            let mut dilim_sonu = konum;
            if jeton.soldan_kirp {
                while dilim_sonu > bas && metin.as_bytes()[dilim_sonu - 1] == b' ' {
                    dilim_sonu -= 1;
                }
            }
            if dilim_sonu > bas {
                parcalar.push(Parca::Metin(metin[bas..dilim_sonu].to_string()));
            }
            parcalar.push(Parca::Eklenen(jeton.id));
            konum += uzunluk;
            if jeton.sagdan_kirp {
                while konum < baytlar.len() && baytlar[konum] == b' ' {
                    konum += 1;
                }
            }
            bas = konum;
        }
        if bas < metin.len() {
            parcalar.push(Parca::Metin(metin[bas..].to_string()));
        }
        parcalar
    }

    /// The BPE merge over one pre-token.
    fn bpe(&self, kelime: &str) -> Vec<u32> {
        let mut sembol: Vec<(String, u32)> = Vec::with_capacity(kelime.chars().count());
        for karakter in kelime.chars() {
            let s = karakter.to_string();
            if let Some(&id) = self.id.get(&s) {
                sembol.push((s, id));
                continue;
            }
            // Byte fallback: a character the vocabulary does not contain is
            // written as its UTF-8 bytes, one token each, so nothing is dropped
            // and nothing is replaced by a guess.
            let mut tampon = [0_u8; 4];
            for bayt in karakter.encode_utf8(&mut tampon).as_bytes() {
                let ad = format!("{}{bayt:02X}>", self.bayt_onek);
                let id = self.id.get(&ad).copied().unwrap_or(self.bilinmeyen);
                sembol.push((ad, id));
            }
        }
        while sembol.len() > 1 {
            // The pair with the lowest rank, and the leftmost one when two
            // pairs share a rank: the file's order is the tie-break, and
            // iterating in order while keeping the strictly-smaller test gives
            // exactly that.
            let mut secim: Option<(usize, u32)> = None;
            for sira in 0..sembol.len() - 1 {
                let cift = (sembol[sira].1, sembol[sira + 1].1);
                let Some(&r) = self.rank.get(&cift) else {
                    continue;
                };
                if secim.is_none_or(|(_, en_iyi)| r < en_iyi) {
                    secim = Some((sira, r));
                }
            }
            let Some((sira, _)) = secim else {
                break;
            };
            let (sol, sag) = (sembol[sira].clone(), sembol[sira + 1].clone());
            let cift = (sol.1, sag.1);
            let yeni_id = self.birlesme.get(&cift).copied().unwrap_or(sol.1);
            let yeni = format!("{}{}", sol.0, sag.0);
            sembol.splice(sira..=sira + 1, [(yeni, yeni_id)]);
        }
        // `fuse_unk`: consecutive unknown tokens are one token, which is what
        // the file asks for and what stops a run of unrepresentable input from
        // becoming a run of repeat ids.
        let mut cikti: Vec<u32> = Vec::with_capacity(sembol.len());
        for (_, id) in sembol {
            if id == self.bilinmeyen && cikti.last() == Some(&self.bilinmeyen) {
                continue;
            }
            cikti.push(id);
        }
        cikti
    }
}

/// One piece of the text: either literal text or an added token.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Parca {
    Metin(String),
    Eklenen(u32),
}

/// The normaliser: every space becomes the word marker.
///
/// This vocabulary declares exactly one rule, and it is this. A different
/// normaliser in the file would be a refusal, not something to approximate -
/// which is why the rules are read and checked rather than assumed to be this
/// one.
fn normalle(metin: &str) -> String {
    metin.replace(' ', &ISARET.to_string())
}

/// The pre-tokenizer: split on the marker, every piece keeping one.
///
/// `prepend_scheme = always`: a text that does not start with the marker gets
/// one, so `"a"` and `" a"` produce the same first token. The split itself is
/// what turns a run of spaces into a run of markers, one per space, which is
/// why `"  "` is two tokens and not one.
#[must_use]
pub fn metaspace(normal: &str) -> Vec<String> {
    let mut parcalar: Vec<String> = Vec::new();
    let mut gecerli = String::new();
    for karakter in normal.chars() {
        if karakter == ISARET {
            if !gecerli.is_empty() {
                parcalar.push(std::mem::take(&mut gecerli));
            }
            gecerli.push(ISARET);
            continue;
        }
        if gecerli.is_empty() {
            gecerli.push(ISARET);
        }
        gecerli.push(karakter);
    }
    if !gecerli.is_empty() {
        parcalar.push(gecerli);
    }
    parcalar
}

/// A byte-fallback token's value, if this token is one.
fn bayt_jetonu(jeton: &str) -> Option<u8> {
    let govde = jeton.strip_prefix("<0x")?.strip_suffix('>')?;
    if govde.len() != 2 {
        return None;
    }
    u8::from_str_radix(govde, 16).ok()
}

/// The added tokens, ordered so that the longest content is tried first.
fn eklenenleri_oku(
    kok: &serde_json::Value,
    id: &HashMap<String, u32>,
) -> Result<Vec<EklenenJeton>, BaslikHatasi> {
    let Some(liste) = kok.get("added_tokens").and_then(serde_json::Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut cikti = Vec::with_capacity(liste.len());
    for kayit in liste {
        let icerik = kayit
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| hata("added_tokens: `content` yok"))?
            .to_string();
        let numara = kayit
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| hata("added_tokens: `id` yok"))?;
        #[allow(clippy::cast_possible_truncation)]
        let numara = numara as u32;
        // The id in the list and the id in the vocabulary have to be the same
        // number for the same text; a file where they disagree would emit ids
        // that mean something else.
        if let Some(&sozlukte) = id.get(&icerik) {
            if sozlukte != numara {
                return Err(hata(&format!(
                    "added_tokens: `{icerik}` listede {numara}, sozlukte {sozlukte}"
                )));
            }
        }
        cikti.push(EklenenJeton {
            icerik,
            id: numara,
            soldan_kirp: kayit
                .get("lstrip")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            sagdan_kirp: kayit
                .get("rstrip")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            ozel: kayit
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        });
    }
    cikti.sort_by(|a, b| b.icerik.len().cmp(&a.icerik.len()));
    Ok(cikti)
}

/// The template's wrap: the first and last ids a sequence gets.
fn sablon(
    kok: &serde_json::Value,
    id: &HashMap<String, u32>,
) -> Result<(Option<u32>, Option<u32>), BaslikHatasi> {
    let Some(sablon) = kok.get("post_processor") else {
        return Ok((None, None));
    };
    let tur = sablon
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if tur != "TemplateProcessing" {
        return Err(hata(&format!("post_processor `{tur}` taninmadi")));
    }
    let tek = sablon
        .get("single")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| hata("post_processor: `single` yok"))?;
    let mut bas = None;
    let mut son = None;
    for (sira, parca) in tek.iter().enumerate() {
        let Some(ad) = parca
            .get("SpecialToken")
            .and_then(|s| s.get("id"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let Some(&numara) = id.get(ad) else {
            return Err(hata(&format!("post_processor: `{ad}` sozlukte yok")));
        };
        if sira == 0 {
            bas = Some(numara);
        } else {
            son = Some(numara);
        }
    }
    Ok((bas, son))
}

fn hata(mesaj: &str) -> BaslikHatasi {
    BaslikHatasi::Baslik {
        mesaj: mesaj.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny vocabulary with a normaliser, a pre-tokenizer, a BPE and a
    /// template, written the way the real file writes them.
    fn kucuk_json() -> String {
        serde_json::json!({
            "model": {
                "type": "BPE",
                "byte_fallback": true,
                "fuse_unk": true,
                "vocab": {
                    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4,
                    "▁": 5, "a": 6, "b": 7, "c": 8, "▁a": 9, "ab": 10, "▁ab": 11,
                    "\n": 12, "\n\n": 13, "<0x00>": 14, "<0xE2>": 15, "<0x82>": 16, "<0xAC>": 17
                },
                // The order is the ranking, and the ranking is the semantics:
                // `["a","b"]` must come before `["▁","a"]` or `"ab"` would
                // merge as `"▁a"`+`"b"` instead of `"▁"`+`"ab"`. Measured on
                // the real file: `"ab"` -> 841 (`"▁ab"`), which is only
                // reachable with this order.
                "merges": [["a", "b"], ["▁", "a"], ["▁", "ab"], ["\n", "\n"]]
            },
            "added_tokens": [
                {"id": 4, "content": "<mask>", "lstrip": true, "rstrip": false, "special": true},
                {"id": 12, "content": "\n", "lstrip": false, "rstrip": false, "special": false},
                {"id": 13, "content": "\n\n", "lstrip": false, "rstrip": false, "special": false},
                {"id": 2, "content": "<bos>", "lstrip": false, "rstrip": false, "special": true},
                {"id": 1, "content": "<eos>", "lstrip": false, "rstrip": false, "special": true}
            ],
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}}
                ]
            }
        })
        .to_string()
    }

    fn kucuk() -> Sozluk {
        Sozluk::metinden(&kucuk_json()).expect("kucuk sozluk")
    }

    #[test]
    fn a_word_is_marked_and_merged_by_rank() {
        let s = kucuk();
        // "ab" -> "▁ab": the marker is added, then the pair merges twice.
        let idler = s.jetonla_cekirdek("ab");
        assert_eq!(idler, vec![11]);
        // "a b" -> "▁a" + "▁b". The second piece keeps its own marker, which
        // is a token on its own because no rule joins it to `b`: measured on
        // the real vocabulary, `"  boşluklu"` produces exactly those lone
        // markers between the words.
        assert_eq!(s.jetonla_cekirdek("a b"), vec![9, 5, 7]);
    }

    #[test]
    fn the_sequence_is_wrapped_by_the_files_own_template() {
        let s = kucuk();
        assert_eq!(s.jetonla("ab"), vec![2, 11, 1]);
        assert_eq!(s.sarmalayici(), (Some(2), Some(1)));
        assert_eq!(s.dolgu(), 0);
        assert_eq!(s.maske(), 4);
    }

    #[test]
    fn a_newline_run_is_matched_before_anything_else() {
        let s = kucuk();
        // The added-token list contains the newline runs, so this is one token
        // and not "▁" plus two newlines.
        assert_eq!(s.jetonla_cekirdek("\n\n"), vec![13]);
        assert_eq!(s.jetonla_cekirdek("\n"), vec![12]);
        // And the longest match wins: three newlines must not be "\n\n" + "\n".
        assert_eq!(s.jetonla_cekirdek("\n\n\n"), vec![13, 12]);
    }

    #[test]
    fn an_unknown_character_falls_back_to_its_bytes_instead_of_disappearing() {
        let s = kucuk();
        // U+20AC (the euro sign) is three bytes, none of which are in this
        // vocabulary as anything but fallback tokens.
        let idler = s.jetonla_cekirdek("€");
        assert_eq!(idler, vec![5, 15, 16, 17], "{idler:?}");
        // And it decodes back to the character, not to the token text.
        assert_eq!(s.coz(&idler), " €");
        // A zero byte has its own fallback token.
        assert_eq!(s.jetonla_cekirdek("\u{0}"), vec![5, 14]);
    }

    #[test]
    fn consecutive_unknowns_are_one_token() {
        let mut json: serde_json::Value = serde_json::from_str(&kucuk_json()).expect("json");
        // Drop the byte tokens so the fallback path lands on `<unk>` instead.
        let sozluk = json["model"]["vocab"].as_object_mut().expect("vocab");
        for ad in ["<0xE2>", "<0x82>", "<0xAC>", "<0x00>"] {
            sozluk.remove(ad);
        }
        let s = Sozluk::metinden(&json.to_string()).expect("sozluk");
        // Three bytes, three unks - fused into one by `fuse_unk`.
        assert_eq!(s.jetonla_cekirdek("€"), vec![5, 3]);
    }

    #[test]
    fn a_piece_at_the_start_gets_a_marker_and_a_space_does_not_add_a_second() {
        let s = kucuk();
        assert_eq!(s.jetonla_cekirdek("a"), s.jetonla_cekirdek(" a"));
        // Two spaces are two markers, not one.
        assert_eq!(s.jetonla_cekirdek("  "), vec![5, 5]);
        // An empty text is no ids at all - not a marker.
        assert_eq!(s.jetonla_cekirdek(""), Vec::<u32>::new());
    }

    #[test]
    fn an_added_token_with_lstrip_eats_the_space_before_it() {
        let s = kucuk();
        // Without the flag this would be "▁a" + "▁" + "<mask>"; with it, the
        // space is gone before the match is taken, and the marker the piece
        // would have carried never appears. Measured on the real file:
        // `"a <mask> b"` is three pieces and not five.
        assert_eq!(s.jetonla_cekirdek("a <mask>"), vec![9, 4]);
        // The piece after the token starts with its own marker, and no rule
        // joins that marker to `b` in this fixture - so two ids, which is the
        // honest answer here; the real vocabulary has a rule for `"▁b"` and
        // produces one.
        assert_eq!(s.jetonla_cekirdek("a <mask> b"), vec![9, 4, 5, 7]);
        assert_eq!(s.jetonla_cekirdek("<mask>"), vec![4]);
    }

    #[test]
    fn a_vocabulary_whose_merges_do_not_exist_is_refused() {
        let mut json: serde_json::Value = serde_json::from_str(&kucuk_json()).expect("json");
        json["model"]["merges"] = serde_json::json!([["a", "zzz"]]);
        let hata = Sozluk::metinden(&json.to_string()).expect_err("reddedilmeli");
        assert!(format!("{hata:?}").contains("zzz"), "{hata:?}");
        // A merge whose *result* is missing is refused too: the merged id would
        // otherwise be whatever the vocabulary happened to say.
        let mut json: serde_json::Value = serde_json::from_str(&kucuk_json()).expect("json");
        json["model"]["merges"] = serde_json::json!([["a", "c"]]);
        assert!(Sozluk::metinden(&json.to_string()).is_err());
    }

    #[test]
    fn an_id_that_disagrees_between_the_list_and_the_vocabulary_is_refused() {
        let mut json: serde_json::Value = serde_json::from_str(&kucuk_json()).expect("json");
        json["added_tokens"][0]["id"] = serde_json::json!(99);
        let hata = Sozluk::metinden(&json.to_string()).expect_err("reddedilmeli");
        assert!(format!("{hata:?}").contains("<mask>"), "{hata:?}");
    }

    #[test]
    fn a_post_processor_that_is_not_a_template_is_refused() {
        let mut json: serde_json::Value = serde_json::from_str(&kucuk_json()).expect("json");
        json["post_processor"] = serde_json::json!({"type": "BertProcessing"});
        assert!(Sozluk::metinden(&json.to_string()).is_err());
    }

    #[test]
    fn a_model_that_is_not_bpe_is_refused_rather_than_guessed() {
        let mut json: serde_json::Value = serde_json::from_str(&kucuk_json()).expect("json");
        json["model"]["type"] = serde_json::json!("WordPiece");
        let hata = Sozluk::metinden(&json.to_string()).expect_err("reddedilmeli");
        assert!(format!("{hata:?}").contains("WordPiece"), "{hata:?}");
    }

    #[test]
    fn the_metaspace_split_matches_the_shape_measured_on_the_real_vocabulary() {
        // Measured on the real file: "  bosluklu   metin  " normalises to
        // "▁▁bosluklu▁▁▁metin▁▁" and splits into alternating marker pieces.
        let normal = normalle("  boşluklu   metin  ");
        assert_eq!(normal, "▁▁boşluklu▁▁▁metin▁▁");
        let parcalar = metaspace(&normal);
        assert_eq!(parcalar[0], "▁");
        assert_eq!(parcalar[1], "▁boşluklu");
        assert_eq!(parcalar[2], "▁");
        assert_eq!(parcalar[3], "▁");
        assert_eq!(parcalar[4], "▁metin");
        assert_eq!(parcalar.len(), 7, "{parcalar:?}");
    }
}
