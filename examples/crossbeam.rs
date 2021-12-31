use crossbeam_channel::unbounded;
use std::{thread, convert::TryFrom};
use tch::{Tensor, nn::{self, Module}, Device};

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

    let tensor1 = Tensor::try_from(vec![1., 2., 3.]).unwrap().internal_cast_float(false);
    let vs1 = nn::VarStore::new(Device::Cpu);
    let model1 = nn::seq()
        .add(nn::linear(&vs1.root() / "layer1", 3, 8, Default::default()))
        .add(nn::linear(&vs1.root() / "layer2", 8, 2, Default::default()));
    let mut vs2 = nn::VarStore::new(Device::Cpu);
    let model2 = nn::seq()
        .add(nn::linear(&vs2.root() / "layer1", 3, 8, Default::default()))
        .add(nn::linear(&vs2.root() / "layer2", 8, 2, Default::default()));

    model1.forward(&tensor1).print();
    model2.forward(&tensor1).print();

    let (s, r) = unbounded();
    thread::spawn(move || s.send(vs1).unwrap());
    let vs3 = r.recv().unwrap();
    vs2.copy(&vs3).unwrap();

    model2.forward(&tensor1).print();
}
