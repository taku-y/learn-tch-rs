use std::borrow::Borrow;
use tch::{Tensor, nn};

fn main() {
    println!("=== Create scalar variable ===");
    let var_store = nn::VarStore::new(tch::Device::Cpu);
    let path = &var_store.root();
    let scalar = path.borrow().var("scalar", &[1], nn::Init::Const(3.0));
    let tensor = Tensor::of_slice(&[2, 3, 4, 5, 6, 7]).reshape(&[2, 3]);

    println!("=== scalar ===");
    scalar.print();
    println!("=== tensor ===");
    tensor.print();
    println!("=== scalar * tensor ===");
    (scalar * tensor).print();

    // === Create scalar variable ===
    // === scalar ===
    //  3
    // [ CPUFloatType{1} ]
    // === tensor ===
    //  2  3  4
    //  5  6  7
    // [ CPUIntType{2,3} ]
    // === scalar * tensor ===
    //   6   9  12
    //  15  18  21
    // [ CPUFloatType{2,3} ]
}
