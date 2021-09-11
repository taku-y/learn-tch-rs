use crossbeam_channel::unbounded;
use std::{thread, convert::TryFrom};
use tch::Tensor;

fn main() {
    let (s1, r) = unbounded();
    let s2 = s1.clone();
    let tensor1 = Tensor::try_from(vec![1., 2., 3.]).unwrap();
    let tensor2 = Tensor::try_from(vec![4., 5., 6.]).unwrap();

    thread::spawn(move || s1.send(tensor1).unwrap());
    thread::spawn(move || s2.send(tensor2).unwrap());

    let msg1 = r.recv().unwrap();
    msg1.print();
    let msg2 = r.recv().unwrap();
    msg2.print();
}
