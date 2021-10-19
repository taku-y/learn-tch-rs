use tch::Tensor;

fn main() {
    println!("=== t1 ===");
    let t1 = Tensor::of_slice(&[3.0f32, 1e-10, 1e-20, 1e-30, 1e-40, 1e-100]);
    t1.print();

    println!("=== t1.log() ===");
    t1.log().print()

    // === t1 ===
    //  3.0000
    //  0.0000
    //  0.0000
    //  0.0000
    //  0.0000
    //  0.0000
    // [ CPUFloatType{6} ]
    // === t1.log() ===
    //  1.0986
    // -23.0259
    // -46.0517
    // -69.0776
    // -92.1034
    //    -inf
    // [ CPUFloatType{6} ]
}
