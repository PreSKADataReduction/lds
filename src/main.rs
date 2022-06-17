use phased_array::{
    cfg::StationCfg
    , station::Station
};

use rsdsp::{
    oscillator::CFreqScanner
};

use ndarray_npy::{
    write_npy
};

use ndarray::{
    Array1
    , ArrayView1
};

use rand::{
    thread_rng
    , Rng
};

use rand_distr::{
    StandardNormal
};


use serde_yaml::from_reader;

fn main() {
    let station_cfg:StationCfg=from_reader(std::fs::File::open("station.yaml").unwrap()).unwrap();
    println!("{:?}", station_cfg);
    let mut station=Station::<f64,f64>::from_cfg(&station_cfg);
    //println!("{:?}", station.fine_ch_freq_in_fs());
    let a=station.coarse_freq_of_fine_ch_in_fs();
    let b=station.fine_ch_freq_in_fs();
    write_npy("a.npy", &ArrayView1::from(&a));
    write_npy("b.npy", &ArrayView1::from(&b));
}
