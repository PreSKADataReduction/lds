use ndarray::{Array2, ArrayView1, ScalarOperand};

use rayon::prelude::*;


use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use rsdsp::{
    frac_delayer::{FracDelayer, ToDelayValue}
    ,ospfb::Analyzer
    ,windowed_fir::pfb_coeff
    , frac_delayer::cfg2delayer
};

use crate::{
    constants::light_speed
    , utils::{angle2xyz, apply_delay}
    , cfg::StationCfg
};

use rustfft::FftNum;

#[derive(Debug)]
pub struct Antenna<T>
where T:std::fmt::Debug
{
    pos: [T; 3],
    channelizer: Analyzer<T, T>,
    delayer: FracDelayer<T>,
}

impl<T> Antenna<T>
where
    T: Float
        + ScalarOperand
        + FloatConst
        + NumAssign
        + std::iter::Sum
        + std::marker::Send
        + std::marker::Sync
        + FftNum
        + Default
        + std::fmt::Debug,
    Complex<T>: ScalarOperand,
{
    pub fn new(
        pos: [T; 3],
        ncoarse_ch: usize,
        coeff: ArrayView1<T>,
        delayer: FracDelayer<T>,
    ) -> Self {
        let channelizer = Analyzer::new(ncoarse_ch, coeff);
        Antenna {
            pos,
            channelizer,
            delayer,
        }
    }

    pub fn acquire(&mut self, azimuth: T, zenith: T, signal: &[T], dt: T) -> Array2<Complex<T>> {
        let dc = angle2xyz(azimuth, zenith);
        
        let delay = -dc
            .iter()
            .zip(self.pos.iter())
            .map(|(&x, &y)| x * y)
            .sum::<T>()
            / light_speed::<T>()
            / dt;
        println!("delay = {:?}", delay);
        let delayed_signal = self.delayer.delay(signal, delay);
        self.channelizer.analyze(&delayed_signal)
    }
}

#[derive(Debug)]
pub struct Station<T> 
where T:std::fmt::Debug
{
    ants: Vec<Antenna<T>>,
    //synthesizer: Synthesizer<f64, f64>,
    dt: T,
}

impl<T> Station<T>
where
    T: Float
        + ScalarOperand
        + FloatConst
        + NumAssign
        + std::iter::Sum
        + std::marker::Send
        + std::marker::Sync
        + FftNum
        + Default
        + std::fmt::Debug,
    Complex<T>: ScalarOperand,
{
    pub fn new(pos: &[[T;3]], ncoarse_ch: usize, coeff_stage1: ArrayView1<T>, delayer: FracDelayer<T>, dt: T)->Self{
        let ants:Vec<_>=pos.iter().map(|pos|{
            Antenna::new(pos.clone(),ncoarse_ch, coeff_stage1.clone(), delayer.clone())
        }).collect();

        Station{ants, dt}
    }

    pub fn calc_required_digital_delay(&self, azimuth: T, zenith: T)->Vec<T>{
        let dc=angle2xyz(azimuth, zenith);
        self.ants.iter().map(|ant| dc.iter().zip(ant.pos.iter()).map(|(&x, &y)| x*y).sum::<T>()/light_speed()/self.dt ).collect()
    }

    pub fn acquire(&mut self, azimuth: T, zenith: T, signal: &[T], digital_delay: &[T])->Array2<Complex<T>>{
        let dt=self.dt;

        self.ants.par_iter_mut().zip(digital_delay.par_iter()).map(|(ant, &d)|{
            let mut channelized=ant.acquire(azimuth, zenith, signal, dt);
            apply_delay(&mut channelized, d);
            channelized
        }).reduce_with(|a,b| a+b).unwrap()
        //self.synthesizer.synthesize(result.view())
    }
}

impl Station<f64>{
    pub fn from_cfg(cfg: &StationCfg)->Self{
        let coeff_coarse =
        pfb_coeff::<f64>(cfg.pfb.nch / 2, cfg.pfb.tap_per_ch, cfg.pfb.k).into_raw_vec();
        let delayer=cfg2delayer(&cfg.delayer);
        Station::new(&cfg.pos, cfg.pfb.nch, ArrayView1::from(&coeff_coarse), delayer, cfg.dt)
    }
}
