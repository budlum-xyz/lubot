//! Weight manifests on the storage layer: content address, erasure plan,
//! holder rule.
//!
//! Directive section 7 asks that weights live as content-addressed objects on
//! the storage layer, erasure-coded and named by a manifest id, so a device
//! does not have to hold the whole model. Three facts have to be computed
//! rather than assumed: the address of the bytes, how the bytes split into
//! shards that survive losses, and where those shards may live.
//!
//! Everything here is a pure function over its arguments. Nothing in this
//! module touches a network, a file, or a chain: the placement rule is a rule
//! about *counts*, and it is enforced before any holder is contacted, because
//! a placement that cannot survive one holder's loss is not repairable later.

use crate::{sha256_hex, Refusal};

/// The number of shard addresses the format reserves.
pub const MAX_SHARDS: u32 = 255;

/// The content address of an artefact's bytes.
#[must_use]
pub fn weight_id(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// Erasure-coding parameters: how many data shards and how many parity shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcParams {
    data: u32,
    parity: u32,
}

impl EcParams {
    /// Build a parameter pair.
    ///
    /// # Errors
    /// [`Refusal::BadErasureParams`] when either count is zero, or when the
    /// total exceeds [`MAX_SHARDS`]. A zero data count describes no artefact;
    /// a zero parity count describes a copy, not a code, and the holder rule
    /// below would have nothing to enforce.
    pub fn new(data: u32, parity: u32) -> Result<Self, Refusal> {
        if data == 0 || parity == 0 || data.saturating_add(parity) > MAX_SHARDS {
            return Err(Refusal::BadErasureParams { data, parity });
        }
        Ok(Self { data, parity })
    }

    /// The data shard count.
    #[must_use]
    pub fn data(self) -> u32 {
        self.data
    }

    /// The parity shard count.
    #[must_use]
    pub fn parity(self) -> u32 {
        self.parity
    }

    /// Every shard, data and parity.
    #[must_use]
    pub fn total(self) -> u32 {
        self.data + self.parity
    }

    /// How many shard losses the code survives: the parity count.
    #[must_use]
    pub fn tolerance(self) -> u32 {
        self.parity
    }
}

/// How an artefact splits: the shard size every shard is padded to, and the
/// padding the last data shard carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardPlan {
    params: EcParams,
    shard_size: u64,
    pad_len: u64,
}

/// Plan the split of an artefact of `total_len` bytes.
///
/// # Errors
/// [`Refusal::EmptyArtefact`] when there are no bytes to address: an empty
/// object has no content address worth publishing, and a plan for it would be
/// a plan for nothing.
pub fn plan(params: EcParams, total_len: u64) -> Result<ShardPlan, Refusal> {
    if total_len == 0 {
        return Err(Refusal::EmptyArtefact);
    }
    let shard_size = total_len.div_ceil(u64::from(params.data));
    let pad_len = shard_size * u64::from(params.data) - total_len;
    Ok(ShardPlan {
        params,
        shard_size,
        pad_len,
    })
}

impl ShardPlan {
    /// The parameters this plan came from.
    #[must_use]
    pub fn params(self) -> EcParams {
        self.params
    }

    /// The size every shard is padded to.
    #[must_use]
    pub fn shard_size(self) -> u64 {
        self.shard_size
    }

    /// The zero padding the data shards carry in total.
    #[must_use]
    pub fn pad_len(self) -> u64 {
        self.pad_len
    }

    /// How many data shards the artefact is cut into.
    #[must_use]
    pub fn data_shards(self) -> u32 {
        self.params.data
    }

    /// How many shards exist in total.
    #[must_use]
    pub fn total_shards(self) -> u32 {
        self.params.total()
    }
}

/// The canonical bytes of one data shard: its slice, zero-padded to the shard
/// size.
///
/// Every shard is the same length, so a shard digest is taken over a length
/// that does not depend on which shard it is - and reassembly can therefore be
/// verified by concatenating data shards and stripping the declared padding.
///
/// # Errors
/// [`Refusal::ShardOutOfRange`] when `index` is not a data shard.
pub fn data_shard(bytes: &[u8], shard_plan: ShardPlan, index: u32) -> Result<Vec<u8>, Refusal> {
    let count = shard_plan.data_shards();
    if index >= count {
        return Err(Refusal::ShardOutOfRange { index, data: count });
    }
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let start = u64::from(index) * shard_plan.shard_size;
    let begin = usize::try_from(start.min(len)).unwrap_or(0);
    let end = usize::try_from(start.saturating_add(shard_plan.shard_size).min(len)).unwrap_or(0);
    let mut shard = Vec::with_capacity(usize::try_from(shard_plan.shard_size).unwrap_or(0));
    shard.extend_from_slice(bytes.get(begin..end).unwrap_or_default());
    while u64::try_from(shard.len()).unwrap_or(u64::MAX) < shard_plan.shard_size {
        shard.push(0);
    }
    Ok(shard)
}

/// One shard assigned to one holder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// The shard's index.
    pub shard: u32,
    /// The holder that keeps it.
    pub holder: String,
}

/// Place every shard round-robin over the holders, in a deterministic order.
///
/// The rule: no holder may keep more shards than the code tolerates losing.
/// With `data` + `parity` shards and `parity` losses survivable, a holder
/// keeping more than `parity` shards makes one holder's departure fatal - so
/// the placement is refused, before any shard is sent anywhere. The number of
/// holders the rule demands is therefore at least `1 + ceil(data / parity)`.
///
/// # Errors
/// [`Refusal::TooFewHolders`] with fewer than two usable holder names (blank
/// names are not holders; duplicates collapse), and
/// [`Refusal::HolderOverload`] when the round-robin would break the rule.
pub fn place(shard_plan: ShardPlan, holders: &[String]) -> Result<Vec<Placement>, Refusal> {
    let mut names: Vec<String> = holders
        .iter()
        .filter(|holder| !holder.trim().is_empty())
        .cloned()
        .collect();
    names.sort();
    names.dedup();
    if names.len() < 2 {
        return Err(Refusal::TooFewHolders(names.len()));
    }
    let total = shard_plan.total_shards();
    let tolerance = shard_plan.params.tolerance();
    let count = u64::try_from(names.len()).unwrap_or(1);
    let per_holder = u64::from(total).div_ceil(count);
    if per_holder > u64::from(tolerance) {
        let holder = names.first().cloned().unwrap_or_default();
        return Err(Refusal::HolderOverload {
            holder,
            shards: u32::try_from(per_holder).unwrap_or(u32::MAX),
            tolerance,
        });
    }
    let mut placements = Vec::with_capacity(usize::try_from(total).unwrap_or(0));
    for shard in 0..total {
        let index = usize::try_from(shard).unwrap_or(0) % names.len();
        placements.push(Placement {
            shard,
            holder: names.get(index).cloned().unwrap_or_default(),
        });
    }
    Ok(placements)
}

#[cfg(test)]
mod tests {
    use super::{data_shard, place, plan, weight_id, EcParams, MAX_SHARDS};
    use crate::sha256_hex;

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn the_address_of_bytes_is_their_digest() {
        assert_eq!(weight_id(b"lubot"), sha256_hex(b"lubot"));
        assert_ne!(weight_id(b"lubot"), weight_id(b"lubot "));
    }

    #[test]
    fn erasure_parameters_are_closed_at_both_ends() {
        assert!(EcParams::new(0, 1).is_err());
        assert!(EcParams::new(1, 0).is_err());
        assert!(EcParams::new(0, 0).is_err());
        assert!(EcParams::new(MAX_SHARDS, 1).is_err());
        let params = EcParams::new(4, 2).expect("valid");
        assert_eq!(params.total(), 6);
        assert_eq!(params.tolerance(), 2);
        assert_eq!(params.data(), 4);
        assert_eq!(params.parity(), 2);
        assert!(EcParams::new(253, 2).is_ok());
    }

    #[test]
    fn an_empty_artefact_has_no_plan() {
        let params = EcParams::new(2, 1).expect("valid");
        assert!(plan(params, 0).is_err());
    }

    #[test]
    fn padding_is_declared_rather_than_hidden() {
        let params = EcParams::new(4, 2).expect("valid");
        let even = plan(params, 12).expect("plan");
        assert_eq!(even.shard_size(), 3);
        assert_eq!(even.pad_len(), 0);
        let odd = plan(params, 10).expect("plan");
        assert_eq!(odd.shard_size(), 3);
        assert_eq!(odd.pad_len(), 2);
        assert_eq!(odd.data_shards(), 4);
        assert_eq!(odd.total_shards(), 6);
    }

    #[test]
    fn data_shards_reassemble_into_the_artefact() {
        let params = EcParams::new(4, 2).expect("valid");
        let bytes: Vec<u8> = (0u8..10).collect();
        let shard_plan = plan(params, 10).expect("plan");
        let mut joined = Vec::new();
        for index in 0..shard_plan.data_shards() {
            joined.extend(data_shard(&bytes, shard_plan, index).expect("shard"));
        }
        let pad = usize::try_from(shard_plan.pad_len()).expect("pad fits");
        joined.truncate(joined.len() - pad);
        assert_eq!(joined, bytes);
        assert_eq!(weight_id(&joined), weight_id(&bytes));
        assert!(data_shard(&bytes, shard_plan, 4).is_err());
    }

    #[test]
    fn a_shard_of_pure_padding_is_still_a_shard() {
        let params = EcParams::new(5, 1).expect("valid");
        let bytes: Vec<u8> = (0u8..12).collect();
        let shard_plan = plan(params, 12).expect("plan");
        assert_eq!(shard_plan.pad_len(), 3);
        let last = data_shard(&bytes, shard_plan, 4).expect("shard");
        assert_eq!(last, vec![0, 0, 0]);
        assert_eq!(
            last.len(),
            usize::try_from(shard_plan.shard_size()).expect("size")
        );
    }

    #[test]
    fn a_placement_that_one_loss_would_destroy_is_refused() {
        let params = EcParams::new(4, 1).expect("valid");
        let shard_plan = plan(params, 100).expect("plan");
        let refusal = place(shard_plan, &names(&["a", "b", "c"])).expect_err("must refuse");
        assert!(refusal.message().contains("would lose the object"));
        let spread = place(shard_plan, &names(&["a", "b", "c", "d", "e"])).expect("place");
        assert_eq!(spread.len(), 5);
    }

    #[test]
    fn one_holder_is_not_a_placement() {
        let params = EcParams::new(1, 1).expect("valid");
        let shard_plan = plan(params, 8).expect("plan");
        assert!(place(shard_plan, &names(&["a"])).is_err());
        assert!(place(shard_plan, &names(&["", "  "])).is_err());
        assert!(place(shard_plan, &[]).is_err());
    }

    #[test]
    fn placement_is_deterministic_and_dedupes() {
        let params = EcParams::new(2, 2).expect("valid");
        let shard_plan = plan(params, 32).expect("plan");
        let forward = place(shard_plan, &names(&["beta", "alpha", "gamma"])).expect("place");
        let backward = place(shard_plan, &names(&["gamma", "beta", "alpha"])).expect("place");
        assert_eq!(forward, backward);
        let with_duplicate =
            place(shard_plan, &names(&["beta", "alpha", "gamma", "beta"])).expect("place");
        assert_eq!(forward, with_duplicate);
        assert_eq!(forward.len(), 4);
        assert_eq!(forward[0].shard, 0);
        assert_eq!(forward[0].holder, "alpha");
    }
}
