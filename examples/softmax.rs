use tch::Tensor;

fn main() {
    let pad = (-f32::MAX as f64) * Tensor::ones(&[3], tch::kind::FLOAT_CPU)
        .internal_cast_float(false);
    let x = Tensor::of_slice(&[1f64, 3., 5., 2.]).internal_cast_float(false);
    let x = Tensor::hstack(&[x, pad]);
    println!("=== x.softmax(0, tch::Kind::Float) ===");
    x.softmax(0, tch::Kind::Float).print();
}
