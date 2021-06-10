use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[0, 0, 0, 0, 0, 0, 0, 0]).reshape(&[4, 2]);
    x.print();

    println!("=== y ===");
    let y = Tensor::of_slice(&[2, 3]);
    y.print();

    println!("=== z ===");
    let z = Tensor::of_slice(&[8]);
    z.print();

    println!("=== x.get(1).copy_(&y) ===");
    x.get(1).copy_(&y);
    x.print();

    println!("=== x.get(2).copy_(&z) ===");
    x.get(2).copy_(&z);
    x.print();

    println!("=== x.get(3).get(1).copy_(&z.get(0)) ===");
    x.get(3).get(1).copy_(&z.get(0));
    x.print();

    // === x ===
    // 0  0
    // 0  0
    // 0  0
    // 0  0
    // [ CPUIntType{4,2} ]
    // === y ===
    // 2
    // 3
    // [ CPUIntType{2} ]
    // === z ===
    // 8
    // [ CPUIntType{1} ]
    // === x.get(1).copy_(&y) ===
    // 0  0
    // 2  3
    // 0  0
    // 0  0
    // [ CPUIntType{4,2} ]
    // === x.get(2).copy_(&z) ===
    // 0  0
    // 2  3
    // 8  8
    // 0  0
    // [ CPUIntType{4,2} ]
    // === x.get(3).get(1).copy_(&z.get(0)) ===
    // 0  0
    // 2  3
    // 8  8
    // 0  8
}
