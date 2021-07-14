use std::convert::{TryFrom, TryInto};
use tch::Tensor;

fn main() {
    let a: Vec<f64> = vec![1., 2., 3., std::f64::consts::PI];
    
    let b = Tensor::try_from(a).unwrap().internal_cast_float(true);
    b.print();

    let c: Vec<f64> = b.try_into().unwrap();
    println!("{:?}", c);

    let d = Tensor::try_from(c).unwrap().internal_cast_float(true);
    d.print();
}