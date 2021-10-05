use tch::Tensor;

fn main() {
    let x = Tensor::of_slice(&[1f64, 2f64, 3f64, -f64::MAX, -f64::MAX]).internal_cast_float(false);
    let logp = x.log_softmax(0, tch::Kind::Float);
    let p = logp.exp();

    x.unsqueeze(0).print();
    logp.unsqueeze(0).print();
    p.unsqueeze(0).print();
    (p * logp).unsqueeze(0).print();

    // 1  2  3 -inf -inf
    // [ CPUFloatType{1,5} ]
    // -2.4076 -1.4076 -0.4076    -inf    -inf
    // [ CPUFloatType{1,5} ]
    //  0.0900  0.2447  0.6652  0.0000  0.0000
    // [ CPUFloatType{1,5} ]
    // -0.2168 -0.3445 -0.2712     nan     nan
    // [ CPUFloatType{1,5} ]
}
