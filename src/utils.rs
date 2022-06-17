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

pub fn clac_efield_gain<T>(f: &[T], fc: &[T], dir: &[T;3], pointing: &[T;3], pos: &[[T;3]], dt: T)->Vec<Complex<T>>
where T:Float+FloatConst+std::iter::Sum
{

    let (nx, n0x): (Vec<T>, Vec<T>)=pos.iter().map(|p|{
        let cdt=crate::constants::light_speed::<T>()*dt;
        let nx=dir.iter().zip(p.iter()).map(|(&a,&b)| a*b).sum::<T>()/cdt;
        let n0x=pointing.iter().zip(p.iter()).map(|(&a,&b)| a*b).sum::<T>()/cdt;
        (nx, n0x)
    }).unzip();
    f.iter().zip(fc.iter()).map(|(&f1, &fc1)|{
        nx.iter().zip(n0x.iter()).map(|(&nx1, &n0x1)| Complex::<T>::new(T::zero(), T::from(2).unwrap()*T::PI()*(f1*nx1-fc1*n0x1)).exp()).sum::<Complex<T>>()
    }).collect()
}


pub fn correct_2stage_error<T>(coarse_channels: &[usize], ncoarse_ch: usize, nfine_per_coarse: usize, delays: &[T])
where T:Float+FloatConst{
    let coase_ch_spacing=T::from(1).unwrap()/T::from(ncoarse_ch).unwrap();
    let coarse_central_freq=coarse_channels.iter()
    .map(|&c| if c<=(ncoarse_ch-1)/2{c as isize} else {(c as isize-ncoarse_ch as isize)})
    .map(|c| {T::from(c).unwrap()*coase_ch_spacing});

    
}

pub fn angle2xyz<T>(azimuth: T, zenith: T)->[T;3]//North is az=0
where T: Float{
    let ca=azimuth.cos();
    let sa=azimuth.sin();
    let cz=zenith.cos();
    let sz=zenith.sin();

    let x=sz*sa;
    let y=sz*ca;
    let z=cz;

    [x,y,z]
}

pub fn dot<T>(x:&[T], y: &[T])->T
where T: Float+std::iter::Sum
{
    x.iter().zip(y.iter()).map(|(&x1,&y1)| x1*y1).sum()
}
