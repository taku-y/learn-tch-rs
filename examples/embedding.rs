use tch::{Tensor, nn::{embedding, Module}};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let emb = embedding(&vs.root(), 10, 128, Default::default());
    let x = emb.forward(&Tensor::of_slice(&[0]));

    println!("=== emb.forward(Tensor::of_slice(&[0])).size()");
    println!("{:?}", x.size());

    // === emb.forward(Tensor::of_slice(&[0])).size()
    // [1, 128]
}
