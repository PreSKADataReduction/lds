use num::traits::Float;

pub fn light_speed<T: Float>() -> T {
    T::from(2.99792458e8).unwrap()
}
