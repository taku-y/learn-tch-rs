use std::{collections::HashMap, iter::FromIterator, convert::TryFrom};
use tch::{Tensor, nn::{self, Module}, Device::Cpu, nn::VarStore};

/// Named tensors to send model parameters using a channel.
pub struct NamedTensors {
    pub named_tensors: HashMap<String, Tensor>,
}

impl NamedTensors {
    /// Copy data of VarStore to CPU.
    pub fn copy_from(vs: &VarStore) -> Self {
        let src = vs.variables();

        tch::no_grad(|| NamedTensors {
            named_tensors: HashMap::from_iter(src.iter().map(|(k, v)| {
                let v = v.detach().to(Cpu).data();
                (k.clone(), v)
            })),
        })
    }

    /// Copy named tensors to [VarStore].
    pub fn copy_to(&self, vs: &mut VarStore) {
        let src = &self.named_tensors;
        let dest = &mut vs.variables();
        debug_assert_eq!(src.len(), dest.len());
    
        tch::no_grad(|| {
            for (name, src) in src.iter() {
                let dest = dest.get_mut(name).unwrap();
                dest.copy_(src);
            }
        });    
    }
}

fn main() {
    let tensor1 = Tensor::try_from(vec![1., 2., 3.]).unwrap().internal_cast_float(false);
    let vs1 = nn::VarStore::new(Cpu);
    let model1 = nn::seq()
        .add(nn::linear(&vs1.root() / "layer1", 3, 8, Default::default()))
        .add(nn::linear(&vs1.root() / "layer2", 8, 2, Default::default()));
    let mut vs2 = nn::VarStore::new(tch::Device::cuda_if_available());
    let model2 = nn::seq()
        .add(nn::linear(&vs2.root() / "layer1", 3, 8, Default::default()))
        .add(nn::linear(&vs2.root() / "layer2", 8, 2, Default::default()));

    model1.forward(&tensor1).print();
    model2.forward(&tensor1).print();

    let nt = NamedTensors::copy_from(&vs1);
    nt.copy_to(&mut vs2);

    model2.forward(&tensor1).print();
}
