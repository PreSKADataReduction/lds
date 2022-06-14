use ndarray::{
    Array2
};


use num::{
    complex::{
        Complex
    }
    , traits::{
        Float, FloatConst
    }
};

use rsdsp::{
    utils::fftfreq
};

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

pub fn angle2xyz<T>(azimuth: T, zenith: T)->[T;3]
where T: Float + FloatConst{
    let ca=azimuth.cos();
    let sa=azimuth.sin();
    let cz=zenith.cos();
    let sz=zenith.sin();

    let x=sz*sa;
    let y=sz*ca;
    let z=cz;

    [x,y,z]
}


