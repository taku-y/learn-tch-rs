use tch::Tensor;

fn main() {
    println!("=== x ===");
    let x = Tensor::of_slice(&[6, 3, 9, 1, 2, 9, 7, 3]).reshape(&[4, 2]);
    x.print();

    println!("=== Tensor::hstack(&[&x, &x]) ===");
    Tensor::hstack(&[&x, &x]).print();

    println!("=== Tensor::vstack(&[&x, &x]) ===");
    Tensor::vstack(&[&x, &x]).print();
}
