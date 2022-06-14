use serde::{Deserialize, Serialize};
use rsdsp::{
    cfg::{
        PfbCfg
        , DelayerCfg
    }
};


#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct StationCfg {
    pub dt: f64
    , pub pos: Vec<[f64;3]>
    , pub pfb:PfbCfg
    , pub delayer: DelayerCfg
}

