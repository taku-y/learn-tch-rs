use tch::Tensor;

fn main() {
    println!("=== x, x.size() ===");
    let x = Tensor::of_slice(&[3, 1, 4, 5]);
    x.print();
    println!("{:?}", x.size());
    println!("=== x.unsqueeze(-1).unsqueeze(-1).size() ===");
    let y = x.unsqueeze(-1).unsqueeze(-1);
    println!("{:?}", y.size());

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
