use ndarray::{Array2, ScalarOperand};

use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use std::{iter::Sum, ops::Mul};

use rustfft::FftNum;

use rsdsp::utils::fftfreq;

use crate::station::Station;

pub fn apply_delay<T>(x: &mut Array2<Complex<T>>, d: T)
where
    T: Float + Copy + FloatConst + std::fmt::Debug,
{
    let two = T::one() + T::one();
    let freqs = fftfreq::<T>(x.shape()[0]);

    for (r, k) in x.rows_mut().into_iter().zip(
        freqs
            .into_iter()
            .map(|f| Complex::<T>::new(T::zero(), -two * T::PI() * f * d).exp()),
    ) {
        for x1 in r {
            *x1 = *x1 * k;
        }
    }
}

pub fn get_freq_to_sample<R, T>(station: &Station<R, T>, subdiv: usize)->Vec<T>
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
    Complex<T>: Copy + std::convert::From<R> + Sum + Default + ScalarOperand,
    R: Copy
        + Mul<T, Output = R>
        + Default
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sum
        + Sync
        + Send,
{
    let nfine_per_coarse = station.csp_pfb.nfine_per_coarse() * subdiv;
    let ncoarse_ch = station.ants[0].channelizer.nch_total();
    let coarse_ch_spacing = T::from(1).unwrap() / T::from(ncoarse_ch).unwrap();
    let mut result = vec![];

    for fc in station.coarse_ch_freq_in_fs(&station.csp_pfb.coarse_ch_selected) {
        for f in 0..nfine_per_coarse {
            result.push(
                fc + (T::from(f).unwrap()
                    - T::from(nfine_per_coarse - 1).unwrap() / T::from(2).unwrap())
                    / T::from(nfine_per_coarse).unwrap()
                    * coarse_ch_spacing,
            );
        }
    }
    result
}

pub fn angle2xyz<T>(azimuth: T, zenith: T) -> [T; 3]
//North is az=0
where
    T: Float,
{
    let ca = azimuth.cos();
    let sa = azimuth.sin();
    let cz = zenith.cos();
    let sz = zenith.sin();

    let x = sz * sa;
    let y = sz * ca;
    let z = cz;

    [x, y, z]
}

pub fn dot<T>(x: &[T], y: &[T]) -> T
where
    T: Float + std::iter::Sum,
{
    x.iter().zip(y.iter()).map(|(&x1, &y1)| x1 * y1).sum()
}
