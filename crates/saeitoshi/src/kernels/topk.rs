//! TopK selection entry point.
//!
//! Runtime-dispatches between the v0.2 heap-based partial sort
//! (`topk_heap`) and the legacy full-sort fallback (`scalar`). The
//! dispatch is resolved once via `OnceLock` so per-row cost is one
//! virtual call — no env-var read in the hot path.
//!
//! Set `SAEITOSHI_TOPK=legacy` to force the legacy path for A/B
//! comparison or regression bisection. Without `topk-v2` compiled in,
//! the legacy path is the only option.

use crate::sparsify::TopKScratch;

#[cfg(feature = "topk-v2")]
use std::sync::OnceLock;

type TopkFn = fn(&[f32], usize, &mut TopKScratch);

#[cfg(feature = "topk-v2")]
fn resolve_topk_fn() -> TopkFn {
    if std::env::var("SAEITOSHI_TOPK").as_deref() == Ok("legacy") {
        super::scalar::topk_select
    } else {
        super::topk_heap::topk_select
    }
}

/// Returns the resolved TopK implementation. Resolution is cached for
/// the process lifetime via `OnceLock`.
#[cfg(feature = "topk-v2")]
#[inline]
pub fn dispatch() -> TopkFn {
    static CHOICE: OnceLock<TopkFn> = OnceLock::new();
    *CHOICE.get_or_init(resolve_topk_fn)
}

#[cfg(not(feature = "topk-v2"))]
#[inline]
pub fn dispatch() -> TopkFn {
    super::scalar::topk_select
}

/// Select the k largest entries from `scores` and write them into
/// `scratch.indexed`, sorted by index ascending. Tie-broken by index
/// ascending.
#[inline]
pub fn select_into(scores: &[f32], k: usize, scratch: &mut TopKScratch) {
    dispatch()(scores, k, scratch);
}

/// Name of the currently-resolved TopK backend ("heap" or "legacy").
/// Used by benches and the diagnostic harness.
pub fn backend_name() -> &'static str {
    #[cfg(feature = "topk-v2")]
    {
        if std::ptr::eq(
            dispatch() as *const (),
            super::topk_heap::topk_select as *const (),
        ) {
            return "heap";
        }
    }
    "legacy"
}
