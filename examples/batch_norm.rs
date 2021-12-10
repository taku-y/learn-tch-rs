use tch::{Tensor, nn::ModuleT};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let data = Tensor::of_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.]).reshape(&[2, 4]);
    data.print();
    let bn = tch::nn::batch_norm1d(&vs.root(), 4, Default::default());
    let tmp = bn.forward_t(&data, false);
    tmp.print();

    //  0  1  2  3
    //  4  5  6  7
    // [ CPUFloatType{2,4} ]
    //  0.0000  0.3451  0.7373  2.7945
    //  3.6334  1.7253  2.2119  6.5205
    // [ CPUFloatType{2,4} ]
}
