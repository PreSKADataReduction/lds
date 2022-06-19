use lds::{
    cfg::StationCfg
    , station::Station
};

use ndarray_npy::{
    write_npy
};

use ndarray::{
    ArrayView1
};


use serde_yaml::from_reader;

fn main() {
    let station_cfg:StationCfg=from_reader(std::fs::File::open("station.yaml").unwrap()).unwrap();
    println!("{:?}", station_cfg);
    let station=Station::<f64,f64>::from_cfg(&station_cfg);
    //println!("{:?}", station.fine_ch_freq_in_fs());
    let a=station.coarse_freq_of_fine_ch_in_fs();
    let b=station.fine_ch_freq_in_fs();
    write_npy("a.npy", &ArrayView1::from(&a)).unwrap();
    write_npy("b.npy", &ArrayView1::from(&b)).unwrap();
}
