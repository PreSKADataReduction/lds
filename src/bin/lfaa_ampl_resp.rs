use progress_bar::*;

use lds::{
    cfg::StationCfg, station::Station, station_src::GeneralSrcBuilder, utils::get_freq_to_sample,
};

use std::fs::create_dir_all;

use clap::{Arg, Command};

use rsdsp::oscillator::COscillator;

use ndarray_npy::write_npy;

use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView1, Axis};

use num::complex::Complex;

use serde_yaml::from_reader;

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
            Arg::new("subdiv")
                .short('s')
                .long("subdiv")
                .takes_value(true)
                .value_name("divide fine channels into")
                .default_value("2"),
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

    let station_cfg: StationCfg =
        from_reader(std::fs::File::open(matches.value_of("station_cfg").unwrap()).unwrap())
            .unwrap();
    let station = Station::<Complex<f64>, f64>::from_cfg(&station_cfg);

    let subdiv = matches
        .value_of("subdiv")
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let freq_to_sample = get_freq_to_sample(&station, subdiv);

    let omega_to_sample: Vec<_> = freq_to_sample
        .iter()
        .map(|f| 2.0 * f * std::f64::consts::PI)
        .collect();

    let nfreq = freq_to_sample.len();

    let siglen = matches
        .value_of("siglen")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let niter = matches.value_of("niter").unwrap().parse::<usize>().unwrap();
    let out_dir = std::path::PathBuf::from(matches.value_of("outdir").unwrap());
    create_dir_all(&out_dir).unwrap();

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

    let digital_delay = station.calc_required_digital_delay(az0, ze0);
    println!("{:?}", digital_delay);
    init_progress_bar(nfreq);
    set_progress_bar_action("Computing", Color::Blue, Style::Bold);

    coarse_resp
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(fine_resp.axis_iter_mut(Axis(1)))
        .zip(omega_to_sample.par_iter())
        .enumerate()
        .for_each(|(_i, ((mut coarse_resp1, mut fine_resp1), &omega))| {
            let mut station = Station::<Complex<f64>, f64>::from_cfg(&station_cfg);
            let mut src_builder = GeneralSrcBuilder::new(
                &station,
                az,
                ze,
                station_cfg.delayer.max_delay,
                station_cfg.delayer.half_tap,
            );
            let mut osc = COscillator::new(0.0, omega);
            let mut n = 0;
            //println!("{}/{}", i, nfreq);
            loop {
                let signal: Vec<_> = (0..siglen).map(|_| osc.get()).collect();
                //let channelized = station.ants[0].channelizer.analyze(&signal);
                //let (coarse1, fine1) = station.acquire_fine(az, ze, &signal, &digital_delay);

                let mut src = src_builder.build(&signal);
                let (coarse1, fine1) = station.acquire_fine(&mut src, &digital_delay);

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
                        &fine1
                            .map(|y| y.norm_sqr())
                            .mean_axis(Axis(1))
                            .unwrap()
                            .view(),
                    );
                    break;
                }
            }
            inc_progress_bar();
        });

    finalize_progress_bar();

    println!("Done, dumping results");
    write_npy(out_dir.join("coarse.npy"), &coarse_resp).unwrap();
    write_npy(out_dir.join("fine.npy"), &fine_resp).unwrap();

    let freq_coarse = Array1::from_vec(
        station.coarse_ch_freq_in_fs(&(0..station.ncoarse_ch()).collect::<Vec<_>>()),
    );
    let freq_fine = Array1::from_vec(station.fine_ch_freq_in_fs());

    write_npy(out_dir.join("coarse_freq.npy"), &freq_coarse).unwrap();
    write_npy(out_dir.join("fine_freq.npy"), &freq_fine).unwrap();
    write_npy(out_dir.join("freq.npy"), &ArrayView1::from(&freq_to_sample)).unwrap();
}
