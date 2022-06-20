use rsdsp::cfg::{DelayerCfg, PfbCfg};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct StationCfg {
    pub dt: f64,
    pub pos: Vec<[f64; 3]>,
    pub coarse_pfb: PfbCfg,
    pub delayer: DelayerCfg,
    pub selected_coarse_ch: Vec<(usize, usize)>,
    pub fine_pfb: PfbCfg,
}

impl StationCfg {
    pub fn nselected_coarse_ch(&self) -> usize {
        self.selected_coarse_ch
            .iter()
            .map(|&(cb, ce)| ce - cb)
            .product()
    }

    pub fn total_nfine_ch(&self) -> usize {
        self.fine_pfb.nch * self.nselected_coarse_ch()
    }
}
