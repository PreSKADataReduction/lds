use phased_array::{
    cfg::StationCfg
    , station::Station
};

use ndarray_npy::{
    write_npy
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
    let mut station=Station::from_cfg(&station_cfg);
    let mut rng=thread_rng();
    let x:f64=rng.sample(StandardNormal{});
    let signal:Vec<f64>=(0..65536).map(|_| rng.sample(StandardNormal{})).collect();
    let delay=station.calc_required_digital_delay(0.0, 0.0);
    let y=station.acquire(90.0_f64.to_radians(), 30.0_f64.to_radians(), &signal, &delay);
    write_npy("./a.npy", &y).unwrap();
}
