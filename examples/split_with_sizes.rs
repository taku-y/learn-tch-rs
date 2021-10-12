use tch::Tensor;

fn main() {
    let x = Tensor::of_slice(&[1, 3, 5, 7, 9, 2, 4, 6, 8, 0]);
    let y = x.split_with_sizes(&[2, 1, 4, 3], 0);
    for (i, t) in y.iter().enumerate() {
        println!("{:?}", (i, t));
    }

    // (0, [1, 3])
    // (1, [5])
    // (2, [7, 9, 2, 4])
    // (3, [6, 8, 0])
}
