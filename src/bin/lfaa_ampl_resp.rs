use progress_bar::*;

use phased_array::{cfg::StationCfg, station::Station};

use std::{
    fs::create_dir_all, path::PathBuf
};

use clap::{Arg, Command};

use rsdsp::oscillator::{CFreqScanner, COscillator};

use ndarray_npy::write_npy;

use ndarray::{Array1, Array2, Axis, parallel::prelude::*};

use rand::{thread_rng, Rng};

use rand_distr::StandardNormal;

use num::complex::Complex;

use serde_yaml::from_reader;

use itertools_num::linspace;

fn main() {
    let matches = Command::new("ampl_resp_2stages")
        .arg(
            Arg::new("station_cfg")
                .short('c')
                .long("cfg")
                .takes_value(true)
                .value_name("config file")
                .required(true),
        )
        .arg(
            Arg::new("fmin_MHz")
                .short('f')
                .long("fmin")
                .allow_hyphen_values(true)
                .takes_value(true)
                .value_name("freq in MHz")
                .required(true),
        )
        .arg(
            Arg::new("fmax_MHz")
                .short('F')
                .long("fmax")
                .allow_hyphen_values(true)
                .takes_value(true)
                .value_name("freq in MHz")
                .required(true),
        )
        .arg(
            Arg::new("nfreq")
                .short('n')
                .long("nfreq")
                .takes_value(true)
                .value_name("nfreq")
                .default_value("1024")
                .required(false),
        )
        .arg(
            Arg::new("siglen")
                .short('l')
                .long("siglen")
                .takes_value(true)
                .value_name("signal length in pt")
                .required(false)
                .default_value("65536"),
        )
        .arg(
            Arg::new("niter")
                .short('t')
                .long("niter")
                .takes_value(true)
                .value_name("niter")
                .default_value("2")
                .required(false),
        )
        .arg(
            Arg::new("outdir")
                .short('o')
                .long("out")
                .takes_value(true)
                .value_name("output dir name")
                .required(true),
        )
        .arg(
            Arg::new("azimuth0")
                .short('A')
                .long("az0")
                .takes_value(true)
                .value_name("az0 in deg")
                .required(true),
        )
        .arg(
            Arg::new("zenith0")
                .short('Z')
                .long("ze0")
                .takes_value(true)
                .value_name("ze0 in deg")
                .required(true),
        )
        .arg(
            Arg::new("azimuth")
                .short('a')
                .long("az")
                .takes_value(true)
                .value_name("az in deg")
                .required(true),
        )
        .arg(
            Arg::new("zenith")
                .short('z')
                .long("ze")
                .takes_value(true)
                .value_name("ze in deg")
                .required(true),
        )
        .get_matches();


        /*
    let station_cfg = StationCfg {
        pos: vec![[0., 0., 0.]],..
        from_reader(std::fs::File::open(matches.value_of("station_cfg").unwrap()).unwrap())
            .unwrap()
    }; */

    let station_cfg:StationCfg=from_reader(std::fs::File::open(matches.value_of("station_cfg").unwrap()).unwrap())
    .unwrap();

    let fmin_Hz = matches
        .value_of("fmin_MHz")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        * 1e6;
    let fmax_Hz = matches
        .value_of("fmax_MHz")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        * 1e6;

    let omega_min = fmin_Hz * station_cfg.dt * 2.0 * std::f64::consts::PI;
    let omega_max = fmax_Hz * station_cfg.dt * 2.0 * std::f64::consts::PI;
    println!("{} {}", omega_min, omega_max);
    let nfreq = matches.value_of("nfreq").unwrap().parse::<usize>().unwrap();
    let siglen = matches
        .value_of("siglen")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let niter = matches.value_of("niter").unwrap().parse::<usize>().unwrap();
    let out_dir = std::path::PathBuf::from(matches.value_of("outdir").unwrap());
    create_dir_all(&out_dir).unwrap();

    let bandwidth = (omega_max - omega_min) * std::f64::consts::PI;
    let domega = bandwidth / (nfreq + 1) as f64;
    let omegas = Vec::from(linspace(omega_min, omega_max - domega, nfreq).collect::<Vec<_>>());

    let freqs=Array1::from_iter(linspace(fmin_Hz/1e6, fmax_Hz/1e6, nfreq));

    let az0 = matches
        .value_of("azimuth0")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        .to_radians();
    let ze0 = matches
        .value_of("zenith0")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        .to_radians();

    let az = matches
        .value_of("azimuth")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        .to_radians();
    let ze = matches
        .value_of("zenith")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        .to_radians();

    let mut coarse_resp = Array2::<f64>::zeros((station_cfg.coarse_pfb.nch, nfreq));
    let mut fine_resp = Array2::<f64>::zeros((station_cfg.total_nfine_ch(), nfreq));
    
    
    println!("{:?}", fine_resp.shape());
    
    /*
        let result =
            rsdsp::ampl_resp::ampl_resp(&mut station.ants[0].channelizer, 0.0, 1.0, 512, 65536, 2);
        write_npy("a.npy", &result);
    */
    //let o_max = 1.0;
    //let o_min = 0.0;
    //let n_freq = 1024;

    //let df = (f_max - f_min) / (n_freq - 1) as f64;
    //let mut result = Array2::<f64>::zeros((station_cfg.coarse_pfb.nch, nfreq));
    let station = Station::<Complex<f64>, f64>::from_cfg(&station_cfg);
    let digital_delay = station.calc_required_digital_delay(az0, ze0);

    init_progress_bar(nfreq);
    set_progress_bar_action("Computing", Color::Blue, Style::Bold);

    
    println!("{:?}", digital_delay);
    coarse_resp
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(fine_resp.axis_iter_mut(Axis(1)))
        .zip(omegas.par_iter())
        .enumerate()
        .for_each(|(i,  ( (mut coarse_resp1, mut fine_resp1), &omega))| {
            let mut station = Station::<Complex<f64>, f64>::from_cfg(&station_cfg);
            let mut osc = COscillator::new(0.0, omega);
            let mut n = 0;
            //println!("{}/{}", i, nfreq);
            loop {
                let signal: Vec<_> = (0..siglen).map(|_| osc.get()).collect();
                //let channelized = station.ants[0].channelizer.analyze(&signal);
                let (coarse1, fine1) = station.acquire_fine(az, ze, &signal, &digital_delay);

                n += 1;
                if n == niter {
                    coarse_resp1.assign(
                        &coarse1
                            .map(|y| y.norm_sqr())
                            .mean_axis(Axis(1))
                            .unwrap()
                            .view(),
                    );

                    fine_resp1.assign(
                        &fine1.map(|y| y.norm_sqr())
                        .mean_axis(Axis(1))
                        .unwrap()
                        .view()
                    );
                    break;
                }
            }
            inc_progress_bar();
        });

    println!("Done, dumping results");
    write_npy(out_dir.join("coarse.npy"), &coarse_resp).unwrap();
    write_npy(out_dir.join("fine.npy"), &fine_resp).unwrap();

    let freq_coarse=Array1::from_vec(station.coarse_ch_freq_in_fs(&(0..station.ncoarse_ch()).collect::<Vec<_>>()));
    let freq_fine=Array1::from_vec(station.fine_ch_freq_in_fs());

    write_npy(out_dir.join("coarse_freq.npy"), &freq_coarse).unwrap();
    write_npy(out_dir.join("fine_freq.npy"), &freq_fine).unwrap();
    write_npy(out_dir.join("freq.npy"), &freqs).unwrap();
    /*

    for (i, &omega) in omegas.iter().enumerate() {
        println!("{} {}", i, omega);
        let mut station = Station::<Complex<f64>, f64>::from_cfg(&station_cfg);

        let digital_delay = station.calc_required_digital_delay(az0, ze0);

        let mut sig_gen = COscillator::new(0.0, omega);



        let mut n = 0;
        let (coarse_data, fine_data) = loop {
            let signal: Vec<_> = (0..siglen).map(|_| sig_gen.get()).collect();
            let (coarse_data, fine_data) = station.acquire_fine(0.0, 0.0, &signal, &digital_delay);

            n += 1;
            if n == niter {
                break (coarse_data, fine_data);
            }
        };

        //println!("{:?}", fine_data.shape());
        let coarse_resp1=coarse_data.mean_axis(Axis(1)).unwrap().map(|x| x.norm_sqr());
        let fine_resp1=fine_data.mean_axis(Axis(1)).unwrap().map(|x| x.norm_sqr());
        coarse_resp.column_mut(i).assign(&coarse_resp1.view());
        fine_resp.column_mut(i).assign(&fine_resp1.view());
    }


    write_npy(out_dir.clone().join("coarse.npy"), &coarse_resp).unwrap();

    write_npy(out_dir.join("fine.npy"), &fine_resp).unwrap();
    */
}
