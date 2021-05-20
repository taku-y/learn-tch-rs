use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[6, 3, 9, 1, 2, 9, 7, 3]).reshape(&[4, 2]);
    x.print();

    println!("=== Tensor::hstack(&[&x, &x]) ===");
    Tensor::hstack(&[&x, &x]).print();

    println!("=== Tensor::vstack(&[&x, &x]) ===");
    Tensor::vstack(&[&x, &x]).print();

    println!("=== Tensor::stack(&[&x, &x, &x], 0) ===");
    Tensor::stack(&[&x, &x, &x], 0).print();

    println!("=== Tensor::stack(&[&x, &x, &x], 1) ===");
    Tensor::stack(&[&x, &x, &x], 1).print();

    println!("=== Tensor::stack(&[&x, &x, &x], 2) ===");
    Tensor::stack(&[&x, &x, &x], 2).print();

    //     === x ===
    //     6  3
    //     9  1
    //     2  9
    //     7  3
    //    [ CPUIntType{4,2} ]
    //    === Tensor::hstack(&[&x, &x]) ===
    //     6  3  6  3
    //     9  1  9  1
    //     2  9  2  9
    //     7  3  7  3
    //    [ CPUIntType{4,4} ]
    //    === Tensor::vstack(&[&x, &x]) ===
    //     6  3
    //     9  1
    //     2  9
    //     7  3
    //     6  3
    //     9  1
    //     2  9
    //     7  3
    //    [ CPUIntType{8,2} ]
    //    === Tensor::stack(&[&x, &x, &x], 0) ===
    //    (1,.,.) = 
    //      6  3
    //      9  1
    //      2  9
    //      7  3
    
    //    (2,.,.) = 
    //      6  3
    //      9  1
    //      2  9
    //      7  3
    
    //    (3,.,.) = 
    //      6  3
    //      9  1
    //      2  9
    //      7  3
    //    [ CPUIntType{3,4,2} ]
    //    === Tensor::stack(&[&x, &x, &x], 1) ===
    //    (1,.,.) = 
    //      6  3
    //      6  3
    //      6  3
    
    //    (2,.,.) = 
    //      9  1
    //      9  1
    //      9  1
    
    //    (3,.,.) = 
    //      2  9
    //      2  9
    //      2  9
    
    //    (4,.,.) = 
    //      7  3
    //      7  3
    //      7  3
    //    [ CPUIntType{4,3,2} ]
    //    === Tensor::stack(&[&x, &x, &x], 2) ===
    //    (1,.,.) = 
    //      6  6  6
    //      3  3  3
    
    //    (2,.,.) = 
    //      9  9  9
    //      1  1  1
    
    //    (3,.,.) = 
    //      2  2  2
    //      9  9  9
    
    //    (4,.,.) = 
    //      7  7  7
    //      3  3  3
    //    [ CPUIntType{4,2,3} ]
}
