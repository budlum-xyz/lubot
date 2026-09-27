//! Gerceklesmis ornek modul: yalnizca bir sembolun karsiligi.

/// Kaynaktaki `softmax`un karsiligi: bu sablonda gercek degil, yer tutucu.
pub(crate) fn yumusat(x: &[f32]) -> Vec<f32> {
    x.to_vec()
}

#[cfg(test)]
mod tests {
    use super::yumusat;

    #[test]
    fn bos_girdi_bos_cikti() {
        assert!(yumusat(&[]).is_empty());
    }
}
