use num::{traits::FloatConst, Complex, Float};

use rayon::prelude::*;

use std::{fmt::Debug, iter::Sum};

use crate::{
    constants::light_speed,
    station::{Antenna, Station},
    utils::{angle2xyz, dot},
};
use rsdsp::{frac_delayer::FracDelayer, oscillator::COscillator};

pub trait StationSrc<R, T>
where
    T: std::fmt::Debug + Float,
    R: std::fmt::Debug,
{
    fn get_sig(&mut self, station: &Station<R, T>) -> Vec<Vec<R>>;
}

pub struct SingleTone<T>
where
    T: std::fmt::Debug + Float,
{
    pub osc: COscillator<T>,
    pub phases: Vec<T>,
    pub src_dir: [T; 3],
    pub sig_len: usize,
}

impl<T> SingleTone<T>
where
    T: std::fmt::Debug + Float + FloatConst,
{
    pub fn new<R: std::fmt::Debug>(
        station: &Station<R, T>,
        az: T,
        ze: T,
        omega: T,
        sig_len: usize,
    ) -> Self {
        SingleTone {
            osc: COscillator {
                phi: T::zero(),
                dphi_dpt: omega,
            },
            phases: vec![T::zero(); station.ants.len()],
            src_dir: angle2xyz(az, ze),
            sig_len,
        }
    }
}

impl<T> StationSrc<Complex<T>, T> for SingleTone<T>
where
    T: Debug + Float + std::iter::Sum,
{
    fn get_sig(&mut self, station: &Station<Complex<T>, T>) -> Vec<Vec<Complex<T>>> {
        let signal: Vec<_> = (0..self.sig_len).map(|_| self.osc.get()).collect();
        station
            .ants
            .iter()
            .map(|a| {
                let nx = dot(&a.pos, &self.src_dir) / light_speed() / station.dt;
                let phase_factor = Complex::<T>::new(T::zero(), nx * self.osc.dphi_dpt).exp();
                signal.iter().map(|&x| phase_factor * x).collect::<Vec<_>>()
            })
            .collect()
    }
}

pub struct GeneralSrcBuilder<R, T>
where
    R: Debug,
    T: Debug + Float,
{
    pub delayers: Vec<FracDelayer<T, R>>,
    pub delays: Vec<T>,
}

pub struct GeneralSrc<R>
where
    R: Debug,
{
    pub signal: Vec<Vec<R>>
}



impl<R, T> GeneralSrcBuilder<R, T>
where
    T: Copy
        + Float
        + FloatConst
        + std::ops::MulAssign<T>
        + ndarray::ScalarOperand
        + num::traits::NumAssign
        + std::iter::Sum
        + std::fmt::Debug
        + Sync
        + Send,
    R: Copy
        + std::ops::Add<R, Output = R>
        + std::ops::Mul<R, Output = R>
        + std::ops::Mul<T, Output = R>
        + std::ops::MulAssign<R>
        + ndarray::ScalarOperand
        + num::traits::NumAssign
        + Sum
        + std::fmt::Debug
        + Sync
        + Send,
{
    pub fn new(station: &Station<R, T>, az: T, ze: T, max_delay: usize, half_tap: usize) -> Self {
        let src_dir = angle2xyz(az, ze);
        let delays = station
            .ants
            .iter()
            .map(|a| -dot(&a.pos, &src_dir) / light_speed() / station.dt)
            .collect();
        let delayers = station
            .ants
            .iter()
            .map(|_| FracDelayer::new(max_delay, half_tap))
            .collect();
        Self { delayers, delays }
    }

    pub fn build(&mut self, sig: &[R])->GeneralSrc<R>{
        let signal=self.delayers.iter_mut().zip(self.delays.iter()).map(|(delayer, &delay)|{
            delayer.delay(sig, delay)
        }).collect();
        GeneralSrc{
            signal
        }
    }
}

impl<R, T> StationSrc<R, T> for GeneralSrc<R>
where R: Debug+Clone,
T: Debug+Float{
    fn get_sig(&mut self, _: &Station<R, T>) -> Vec<Vec<R>> {
        self.signal.clone()
    }
}
