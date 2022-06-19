use lds::{cfg::StationCfg, station::Station};

use std::{fs::create_dir_all};

use clap::{Arg, Command};

use ndarray_npy::write_npy;

use ndarray::{Array1};

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

    let station=Station::<Complex<f64>, f64>::from_cfg(&station_cfg);
    
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

    let gain_ideal=Array1::from(station.gain_ideal(az, ze, az0, ze0));
    let gain_2stage=Array1::from(station.gain_2stage(az, ze, az0, ze0));

    write_npy(out_dir.join("gain_ideal.npy"), &gain_ideal).unwrap();
    write_npy(out_dir.join("gain_2stage.npy"), &gain_2stage).unwrap();
}
