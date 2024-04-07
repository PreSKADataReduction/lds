use progress_bar::*;

use lds::{
    cfg::StationCfg, station::Station, station_src::GeneralSrcBuilder, utils::get_freq_to_sample,
};

use std::fs::create_dir_all;

use clap::Parser;

use rsdsp::oscillator::COscillator;

use ndarray_npy::write_npy;

use ndarray::{parallel::prelude::*, Array1, Array2, ArrayView1, Axis};

use num::complex::Complex;

use serde_yaml::from_reader;

type FloatType = f64;

#[derive(Debug, Parser)]
#[clap(author, about, version)]
struct Args {
    #[clap(short('c'), long("cfg"), value_name("config file"))]
    station_cfg: String,

    #[clap(
        short('s'),
        long("subdiv"),
        value_name("divid fine channels into"),
        default_value("2")
    )]
    subdiv: usize,

    #[clap(
        short('l'),
        long("siglen"),
        value_name("signal length in pt"),
        default_value("65536")
    )]
    siglen: usize,

    #[clap(short('t'), long("niter"), value_name("niter"), default_value("2"))]
    niter: usize,

    #[clap(short('o'), long("out"), value_name("output dir name"))]
    outdir: String,

    #[clap(short('A'), long("az0"), value_name("az0 in deg"))]
    azimuth0: FloatType,

    #[clap(short('Z'), long("zenith0"), value_name("ze0 in deg"))]
    zenith0: FloatType,

    #[clap(short('a'), long("az"), value_name("az in deg"))]
    azimuth: FloatType,

    #[clap(short('z'), long("zenith"), value_name("ze in deg"))]
    zenith: FloatType,
}

fn main() {
    let args = Args::parse();
    /*
    let station_cfg = StationCfg {
        pos: vec![[0., 0., 0.]],..
        from_reader(std::fs::File::open(matches.value_of("station_cfg").unwrap()).unwrap())
            .unwrap()
    }; */

    let station_cfg: StationCfg =
        from_reader(std::fs::File::open(args.station_cfg).unwrap()).unwrap();
    let station = Station::<Complex<f64>, f64>::from_cfg(&station_cfg);

    let subdiv = args.subdiv;

    let freq_to_sample = get_freq_to_sample(&station, subdiv);

    let omega_to_sample: Vec<_> = freq_to_sample
        .iter()
        .map(|f| 2.0 * f * std::f64::consts::PI)
        .collect();

    let nfreq = freq_to_sample.len();

    let siglen = args.siglen;
    let niter = args.niter;
    let out_dir = std::path::PathBuf::from(args.outdir);
    create_dir_all(&out_dir).unwrap();

    let az0 = args.azimuth0.to_radians();
    let ze0 = args.zenith0.to_radians();

    let az = args.azimuth.to_radians();
    let ze = args.zenith.to_radians();

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
