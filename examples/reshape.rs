use tch::Tensor;

fn main() {
    let x = Tensor::of_slice(&[1, 3, 5, 2, 6, 9, 1, 2, 0, 8, 3, 4]);
    x.print();

    let y = x.reshape(&[3, 4]);
    y.print();

    let z = y.reshape(&[6, 2]);
    z.print();

    let z2 = z.reshape(&[2, 3, 2]);
    z2.print();

    //     1
    //     3
    //     5
    //     2
    //     6
    //     9
    //     1
    //     2
    //     0
    //     8
    //     3
    //     4
    //    [ CPUIntType{12} ]
    //     1  3  5  2
    //     6  9  1  2
    //     0  8  3  4
    //    [ CPUIntType{3,4} ]
    //     1  3
    //     5  2
    //     6  9
    //     1  2
    //     0  8
    //     3  4
    //    [ CPUIntType{6,2} ]
    // (1,.,.) = 
    //   1  3
    //   5  2
    //   6  9

    // (2,.,.) = 
    //   1  2
    //   0  8
    //   3  4
    // [ CPUIntType{2,3,2} ]
}
