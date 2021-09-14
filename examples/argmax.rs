use tch::Tensor;

fn main() {
    let x = Tensor::of_slice(&[1, 3, 5, 2]);
    println!("=== x.argmax(0, false) ===");
    x.argmax(0, false).print();

    let x = Tensor::of_slice(&[1, 3, 5, 2]);
    println!("=== ix: i64 = x.argmax(0, false).into() ===");
    let ix: i64 = x.argmax(0, false).into();
    println!("{:?}", ix);

    let x = Tensor::of_slice(&[1, 3, 5, 2]).reshape(&[2, 2]);
    println!("=== ix: Vec<i64> = x.argmax(0, false).into() ===");
    let ix: Vec<i64> = x.argmax(0, false).into();
    println!("{:?}", ix);

    // === x.argmax(0, false) ===
    // 2
    // [ CPULongType{} ]
    // === ix: i64 = x.argmax(0, false).into() ===
    // 2
    // === ix: i64 = x.argmax(0, false).into() ===
    // [1, 0]
}
