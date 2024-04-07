use lds::{cfg::StationCfg, station::Station};

use std::fs::create_dir_all;

use clap::Parser;

use ndarray_npy::write_npy;

use ndarray::Array1;

use num::complex::Complex;

use serde_yaml::from_reader;

#[derive(Debug, Parser)]
#[clap(author, version, about)]
struct Args {
    #[clap(short = 'c', long = "cfg", value_name = "station cfg")]
    station_cfg: String,

    #[clap(short = 'o', long = "out", value_name = "outdir")]
    outdir: String,

    #[clap(short = 'A', long = "az0", value_name = "azimuth0")]
    azimuth0: f64,

    #[clap(short = 'Z', long = "ze0", value_name = "zenith0")]
    zenith0: f64,

    #[clap(short = 'a', long = "az", value_name = "azimuth")]
    azimuth: f64,

    #[clap(short = 'z', long = "ze", value_name = "zenith")]
    zenith: f64,
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

    let out_dir = std::path::PathBuf::from(args.outdir);
    create_dir_all(&out_dir).unwrap();

    let az0 = args.azimuth0.to_radians();
    let ze0 = args.zenith0.to_radians();

    let az = args.azimuth.to_radians();
    let ze = args.zenith.to_radians();

    let gain_ideal = Array1::from(station.gain_ideal(az, ze, az0, ze0));
    let gain_2stage = Array1::from(station.gain_2stage(az, ze, az0, ze0));

    write_npy(out_dir.join("gain_ideal.npy"), &gain_ideal).unwrap();
    write_npy(out_dir.join("gain_2stage.npy"), &gain_2stage).unwrap();
}
