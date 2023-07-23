use tch::{Tensor, IndexOp};

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    x.print();

    println!("=== y ===");
    let y = Tensor::of_slice(&[-1, -2, -3, -4, -5, -6, -7, -8]);
    y.print();

    println!("=== z ===");
    let z = Tensor::of_slice(&[-1, -2, -3, -4]).reshape(&[2, 2]);
    z.print();

    println!("=== x.slice(0, 3, 6, 1).copy_(&y.slice(0, 3, 6, 1)) ===");
    x.slice(0, 3, 6, 1).copy_(&y.slice(0, 3, 6, 1));
    x.print();

    println!("=== z.unsqueeze(0).flat_view().squeeze1(0).copy_(&y.slice(0, 3, 7, 1))");
    z.unsqueeze(0).flat_view().squeeze1(0).copy_(&y.slice(0, 3, 7, 1));
    z.print();

    println!("=== x.reshape(&[2, 4])");
    x.reshape(&[2, 4]).print();

    println!("=== x.reshape(&[2, 4]).i((.., 2))");
    x.reshape(&[2, 4]).i((.., 2)).print();

    println!("=== x.reshape(&[2, 2, 2])");
    x.reshape(&[2, 2, 2]).print();

    println!("=== x.reshape(&[2, 2, 2]).i((.., 0, ..))");
    x.reshape(&[2, 2, 2]).i((.., 0, ..)).print();

    //     === x ===
    //     1
    //     2
    //     3
    //     4
    //     5
    //     6
    //     7
    //     8
    //    [ CPUIntType{8} ]
    //    === y ===
    //    -1
    //    -2
    //    -3
    //    -4
    //    -5
    //    -6
    //    -7
    //    -8
    //    [ CPUIntType{8} ]
    //    === z ===
    //    -1 -2
    //    -3 -4
    //    [ CPUIntType{2,2} ]
    //    === x.slice(0, 3, 6, 1).copy_(&y.slice(0, 3, 6, 1)) ===
    //     1
    //     2
    //     3
    //    -4
    //    -5
    //    -6
    //     7
    //     8
    //    [ CPUIntType{8} ]
    //    === z.unsqueeze(0).flat_view().squeeze1(0).copy_(&y.slice(0, 3, 7, 1))
    //    -4 -5
    //    -6 -7
    //    [ CPUIntType{2,2} ]
    //    === x.reshape(&[2, 4])
    //    1  2  3 -4
    //   -5 -6  7  8
    //   [ CPUIntType{2,4} ]
    //   === x.reshape(&[2, 4]).i((.., 2))
    //    3
    //    7
    //   [ CPUIntType{2} ]
    //   === x.reshape(&[2, 2, 2])
    //   (1,.,.) = 
    //    1  2
    //    3 -4
    //
    //   (2,.,.) = 
    //   -5 -6
    //    7  8
    //   [ CPUIntType{2,2,2} ]
    //   === x.reshape(&[2, 2, 2]).i((.., 0, ..))
    //    1  2
    //   -5 -6
    //[ CPUIntType{2,2} ]
}
