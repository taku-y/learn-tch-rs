use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[3, 1, 4, 5, 8, 2, 1, 0, 4, 4]).reshape(&[2, 5]);
    x.print();
    println!("=== x.pow(2).where1(&x.lt(4), &-(&x)) ===");
    let y = x.pow(2).where1(&x.lt(4), &-(&x));
    y.print();

    // === x ===
    //  3  1  4  5  8
    //  2  1  0  4  4
    // [ CPUIntType{2,5} ]
    // === x.pow(2).where1(&x.lt(4), &-(&x)) ===
    //  9  1 -4 -5 -8
    //  4  1  0 -4 -4
    // [ CPUIntType{2,5} ]
}
