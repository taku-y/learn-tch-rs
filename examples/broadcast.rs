use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[3, 1, 4, 5]);
    x.print();
    println!("=== y ===");
    let y = Tensor::of_slice(&[6, 3, 9, 1, 2, 9, 7, 3]).reshape(&[4, 2]);
    y.print();
    println!("=== x.unsqueeze(-1) - y ===");
    (x.unsqueeze(-1) - y).print();

    //     === x ===
    //     3
    //     1
    //     4
    //     5
    //    [ CPUIntType{4} ]
    //    === y ===
    //     6  3
    //     9  1
    //     2  9
    //     7  3
    //    [ CPUIntType{4,2} ]
    //    === x.unsqueeze(-1) - y ===
    //    -3  0
    //    -8  0
    //     2 -5
    //    -2  2
    //    [ CPUIntType{4,2} ]
}
