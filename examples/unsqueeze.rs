use tch::Tensor;

fn main() {
    println!("=== x, x.size() ===");
    let x = Tensor::of_slice(&[3, 1, 4, 5]);
    x.print();
    println!("{:?}", x.size());
    println!("=== x.unsqueeze(-1).unsqueeze(-1).size() ===");
    let y = x.unsqueeze(-1).unsqueeze(-1);
    println!("{:?}", y.size());
    println!("=== x.unsqueeze(0).size() ===");
    let z = x.unsqueeze(0);
    println!("{:?}", z.size());

    // === x, x.size() ===
    //  3
    //  1
    //  4
    //  5
    // [ CPUIntType{4} ]
    // [4]
    // === x.unsqueeze(-1).unsqueeze(-1).size() ===
    // [4, 1, 1]
    // === x.unsqueeze(0).size() ===
    // [1, 4]
}
