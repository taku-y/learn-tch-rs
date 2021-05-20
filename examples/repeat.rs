use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[6, 3, 9, 1, 2, 9, 7, 3]).reshape(&[4, 2]);
    x.print();
    println!("=== x.repeat(&[2, 3]) ===");
    x.repeat(&[2, 3]).print();

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
