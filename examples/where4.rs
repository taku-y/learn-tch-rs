use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[3. as f32, 1., 4., 5., 8., 2., 1., 0., 4., 4.])
        .reshape(&[2, 5]);
    x.print();
    println!("=== Tensor::where4(x.lt(1), 0., 1.) ===");
    let y = Tensor::where4(&x.lt(3), 0., 1.);
    y.print();

    //     === x ===
    //     3  1  4  5  8
    //     2  1  0  4  4
    //    [ CPUFloatType{2,5} ]
    //    === Tensor::where4(x.lt(1), 0., 1.) ===
    //     1  0  1  1  1
    //     0  0  0  1  1
    //    [ CPUFloatType{2,5} ]
}
