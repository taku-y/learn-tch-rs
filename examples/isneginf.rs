use tch::Tensor;

fn main() {
    let x = Tensor::of_slice(&[1f64, 2f64, 3f64, -f64::MAX, -f64::MAX]).internal_cast_float(false);
    let y = x.isneginf();
    let z = y.logical_not();
    let m = &x * &z;

    x.unsqueeze(0).print();
    y.unsqueeze(0).print();
    z.unsqueeze(0).print();
    m.unsqueeze(0).print();

    // 1  2  3 -inf -inf
    // [ CPUFloatType{1,5} ]
    //  0  0  0  1  1
    // [ CPUBoolType{1,5} ]
}
