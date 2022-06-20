use ndarray::{Array2, ArrayView1, ScalarOperand};

use std::{iter::Sum, ops::Mul};

use num::{
    complex::Complex,
    traits::{Float, FloatConst, NumAssign},
};

use rsdsp::{
    csp_pfb::CspPfb, cspfb::Analyzer as CsPfb, ospfb::Analyzer as OsPfb, windowed_fir::pfb_coeff,
};

use crate::{
    cfg::StationCfg,
    constants::light_speed,
    station_src::StationSrc,
    utils::{angle2xyz, apply_delay, dot},
};

use rustfft::FftNum;

#[derive(Debug)]
pub struct Antenna<R, T>
where
    T: std::fmt::Debug + Float,
    R: std::fmt::Debug,
{
    pub pos: [T; 3],
    pub channelizer: OsPfb<R, T>,
    //pub delayer: FracDelayer<T, R>,
}

impl<R, T> Antenna<R, T>
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
    R: Copy
        + Mul<T, Output = R>
        + Default
        + ScalarOperand
        + NumAssign
        + std::fmt::Debug
        + Sum
        + Sync
        + Send,
    Complex<T>: Copy + std::convert::From<R> + Sum + Default + ScalarOperand,
{
    pub fn new(
        pos: [T; 3],
        ncoarse_ch: usize,
        coeff: ArrayView1<T>,
        //delayer: FracDelayer<T, R>,
    ) -> Self {
        let channelizer = OsPfb::new(ncoarse_ch, coeff);
        Antenna {
            pos,
            channelizer,
            //delayer,
        }
    }

    pub fn acquire(&mut self, signal: &[R]) -> Array2<Complex<T>> {
        //let dc = angle2xyz(azimuth, zenith);

        /*
        let delay = -src_dir
            .iter()
            .zip(self.pos.iter())
            .map(|(&x, &y)| x * y)
            .sum::<T>()
            / light_speed::<T>()
            / dt;
        */
        //println!("delay = {:?}", delay);
        //let delayed_signal = self.delayer.delay(signal, delay);
        self.channelizer.analyze(signal)
    }
}

#[derive(Debug)]
pub struct Station<R, T>
where
    T: std::fmt::Debug + Float,
    R: std::fmt::Debug,
{
    pub dt: T,
    pub ants: Vec<Antenna<R, T>>,
    //synthesizer: Synthesizer<f64, f64>,
    pub csp_pfb: CspPfb<T>,
}

impl<R, T> Station<R, T>
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
    pub fn new(
        pos: &[[T; 3]],
        ncoarse_ch: usize,
        coeff_stage1: ArrayView1<T>,
        nfine_ch: usize,
        coeff_stage2: ArrayView1<T>,
        coarse_ch_selected: &[usize],
        dt: T,
    ) -> Self {
        let ants: Vec<_> = pos
            .iter()
            .map(|pos| {
                Antenna::new(
                    *pos,
                    ncoarse_ch,
                    coeff_stage1,
                    //delayer.clone(),
                )
            })
            .collect();

        let fine_pfb = CsPfb::<Complex<T>, T>::new(nfine_ch * 2, coeff_stage2);
        let csp_pfb = CspPfb::new(coarse_ch_selected, &fine_pfb);

        Station { ants, dt, csp_pfb }
    }

    pub fn ncoarse_ch(&self) -> usize {
        self.ants[0].channelizer.nch_total()
    }

    pub fn calc_required_digital_delay(&self, azimuth: T, zenith: T) -> Vec<T> {
        let dc = angle2xyz(azimuth, zenith);
        self.ants
            .iter()
            .map(|ant| {
                dc.iter()
                    .zip(ant.pos.iter())
                    .map(|(&x, &y)| x * y)
                    .sum::<T>()
                    / light_speed()
                    / self.dt
            })
            .collect()
    }

    pub fn coarse_ch_freq_in_fs(&self, ch: &[usize]) -> Vec<T> {
        let ncoarse_ch = self.ants[0].channelizer.nch_total();
        let coase_ch_spacing = T::from(1).unwrap() / T::from(ncoarse_ch).unwrap();
        ch.iter()
            .map(|&c| {
                if c <= (ncoarse_ch - 1) / 2 {
                    c as isize
                } else {
                    c as isize - ncoarse_ch as isize
                }
            })
            .map(|c| T::from(c).unwrap() * coase_ch_spacing)
            .collect()
    }

    pub fn fine_ch_freq_in_fs(&self) -> Vec<T> {
        let nfine_per_coarse = self.csp_pfb.nfine_per_coarse();
        let ncoarse_ch = self.ants[0].channelizer.nch_total();
        let coarse_ch_spacing = T::from(1).unwrap() / T::from(ncoarse_ch).unwrap();
        let mut result = vec![];

        for fc in self.coarse_ch_freq_in_fs(&self.csp_pfb.coarse_ch_selected) {
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

    pub fn coarse_freq_of_fine_ch_in_fs(&self) -> Vec<T> {
        let nfine_per_coarse = self.csp_pfb.nfine_per_coarse();
        let mut result = vec![];
        for fc in self.coarse_ch_freq_in_fs(&self.csp_pfb.coarse_ch_selected) {
            for _ in 0..nfine_per_coarse {
                result.push(fc);
            }
        }
        result
    }

    pub fn gain(&self, f: &[T], fc: &[T], az: T, ze: T, az0: T, ze0: T) -> Vec<Complex<T>> {
        let cdt = light_speed::<T>() * self.dt;
        let n = angle2xyz(az, ze);
        let n0 = angle2xyz(az0, ze0);

        let pos: Vec<_> = self.ants.iter().map(|a| a.pos).collect();

        let (nx, n0x): (Vec<T>, Vec<T>) = pos
            .iter()
            .map(|p| {
                let nx = dot(&n, p) / cdt;
                let n0x = dot(&n0, p) / cdt;
                (nx, n0x)
            })
            .unzip();
        f.iter()
            .zip(fc.iter())
            .map(|(&f1, &fc1)| {
                nx.iter()
                    .zip(n0x.iter())
                    .map(|(&nx1, &n0x1)| {
                        Complex::<T>::new(
                            T::zero(),
                            T::from(2).unwrap() * T::PI() * (f1 * nx1 - fc1 * n0x1),
                        )
                        .exp()
                    })
                    .sum::<Complex<T>>()
            })
            .collect()
    }

    pub fn gain_ideal(&self, az: T, ze: T, az0: T, ze0: T) -> Vec<Complex<T>> {
        let f = self.fine_ch_freq_in_fs();
        self.gain(&f, &f, az, ze, az0, ze0)
    }

    pub fn gain_2stage(&self, az: T, ze: T, az0: T, ze0: T) -> Vec<Complex<T>> {
        let f = self.fine_ch_freq_in_fs();
        let fc = self.coarse_freq_of_fine_ch_in_fs();
        self.gain(&f, &fc, az, ze, az0, ze0)
    }

    pub fn acquire(
        &mut self,
        src: &mut dyn StationSrc<R, T>,
        digital_delay: &[T],
    ) -> Array2<Complex<T>> {
        //let src_dir=angle2xyz(azimuth, zenith);
        let signal = src.get_sig(self);
        self.ants
            .iter_mut()
            .zip(signal.into_iter())
            .zip(digital_delay.iter())
            .map(|((ant, signal1), &d)| {
                let mut channelized = ant.acquire(&signal1);
                apply_delay(&mut channelized, d);
                channelized
            })
            .reduce(|a, b| a + b)
            .unwrap()
        //self.synthesizer.synthesize(result.view())
    }

    pub fn acquire_fine(
        &mut self,
        src: &mut dyn StationSrc<R, T>,
        digital_delay: &[T],
    ) -> (Array2<Complex<T>>, Array2<Complex<T>>) {
        let coarse_data = self.acquire(src, digital_delay);
        let fine_data = self.csp_pfb.analyze(coarse_data.view());
        (coarse_data, fine_data)
    }
}

impl<R, T> Station<R, T>
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
    pub fn from_cfg(cfg: &StationCfg) -> Self {
        let coeff_coarse = pfb_coeff::<T>(
            cfg.coarse_pfb.nch / 2,
            cfg.coarse_pfb.tap_per_ch,
            T::from(cfg.coarse_pfb.k).unwrap(),
        )
        .into_raw_vec();

        let coeff_fine = pfb_coeff::<T>(
            cfg.fine_pfb.nch * 2,
            cfg.fine_pfb.tap_per_ch,
            T::from(cfg.fine_pfb.k).unwrap(),
        )
        .into_raw_vec();

        let mut coarse_ch_selected = Vec::new();
        cfg.selected_coarse_ch.iter().for_each(|&(c1, c2)| {
            (c1..c2).for_each(|c| {
                coarse_ch_selected.push(c);
            });
        });

        assert_eq!(
            cfg.coarse_pfb.tap_per_ch * cfg.coarse_pfb.nch / 2,
            coeff_coarse.len()
        );
        assert_eq!(
            cfg.fine_pfb.tap_per_ch * cfg.fine_pfb.nch * 2,
            coeff_fine.len()
        );

        let pos: Vec<_> = cfg
            .pos
            .iter()
            .map(|x| {
                [
                    T::from(x[0]).unwrap(),
                    T::from(x[1]).unwrap(),
                    T::from(x[2]).unwrap(),
                ]
            })
            .collect();

        Station::new(
            &pos,
            cfg.coarse_pfb.nch,
            ArrayView1::from(&coeff_coarse),
            cfg.fine_pfb.nch,
            ArrayView1::from(&coeff_fine),
            &coarse_ch_selected,
            T::from(cfg.dt).unwrap(),
        )
    }
}
