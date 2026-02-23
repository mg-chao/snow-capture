use crate::backend::CaptureBlitRegion;

/// Spin-poll tuning constants shared by both GPU backends.
pub(crate) const REGION_SPIN_INITIAL_POLLS: u32 = 4;

/// Trait abstracting the slot reset behavior that differs between backends.
///
/// DXGI's `reset_region_pipeline` calls `reset_runtime_state` (preserves GPU
/// textures), while WGC calls `invalidate` (drops them).  Both backends call
/// `invalidate` for the stronger teardown path.
pub(crate) trait RegionSlot: Default {
    /// Soft reset — clear runtime bookkeeping but may keep GPU resources.
    fn soft_reset(&mut self);
    /// Hard reset — drop everything including GPU resources.
    fn hard_reset(&mut self);
}

/// Shared region-pipeline bookkeeping embedded by both GPU backends.
///
/// Owns the staging-slot ring, pending/write indices, adaptive spin count,
/// and the cached blit region.  The actual slot type and ring size are
/// parameterised so each backend keeps its own concrete types.
pub(crate) struct RegionPipelineState<S: RegionSlot, const N: usize> {
    pub slots: [S; N],
    pub pending_slot: Option<usize>,
    pub next_write_slot: usize,
    pub adaptive_spin_polls: u32,
    pub blit: Option<CaptureBlitRegion>,
}

impl<S: RegionSlot, const N: usize> RegionPipelineState<S, N> {
    pub fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| S::default()),
            pending_slot: None,
            next_write_slot: 0,
            adaptive_spin_polls: REGION_SPIN_INITIAL_POLLS,
            blit: None,
        }
    }

    /// Soft-reset the pipeline: clear bookkeeping and soft-reset each slot.
    pub fn reset(&mut self) {
        self.pending_slot = None;
        self.next_write_slot = 0;
        self.adaptive_spin_polls = REGION_SPIN_INITIAL_POLLS;
        self.blit = None;
        for slot in &mut self.slots {
            slot.soft_reset();
        }
    }

    /// Hard-reset the pipeline: clear bookkeeping and fully invalidate each slot.
    pub fn invalidate(&mut self) {
        self.pending_slot = None;
        self.next_write_slot = 0;
        self.adaptive_spin_polls = REGION_SPIN_INITIAL_POLLS;
        self.blit = None;
        for slot in &mut self.slots {
            slot.hard_reset();
        }
    }

    /// Ensure the pipeline is configured for `blit`.  If the blit region
    /// changed, soft-resets the pipeline and stores the new region.
    pub fn ensure_blit(&mut self, blit: CaptureBlitRegion) {
        if self.blit == Some(blit) {
            return;
        }
        self.reset();
        self.blit = Some(blit);
    }
}
