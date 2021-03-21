use tch::Tensor;

fn main() {
    println!("=== Original ===");
    let t1 = Tensor::of_slice(&[3, 1, 4, 5, 8, 2, 1, 0, 4, 4]).reshape(&[2, 5]);
    t1.print();
    let (t2, t3) = t1.min2(0, false);
    println!("=== min2, 1st ret val ===");
    t2.unsqueeze(0).print();
    println!("=== min2, 2nd ret val ===");
    t3.unsqueeze(0).print();

    //     === Original ===
    //     3  1  4  5  8
    //     2  1  0  4  4
    //    [ CPUIntType{2,5} ]
    //    === min2, 1st ret val ===
    //     2  1  0  4  4
    //    [ CPUIntType{1,5} ]
    //    === min2, 2nd ret val ===
    //     1  0  1  1  1
    //    [ CPULongType{1,5} ]
}
