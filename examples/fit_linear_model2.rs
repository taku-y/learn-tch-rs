use std::{convert::TryFrom};
use tch::{Tensor, nn, nn::{Module, OptimizerConfig}};

struct LinearModel {
    vs: nn::VarStore,
    model1: nn::Sequential,
    model2: nn::Sequential,
    opt1: nn::Optimizer<nn::Adam>,
    opt2: nn::Optimizer<nn::Adam>,
}

impl LinearModel {
    pub fn new() -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let p = vs.root();
        let model1 = nn::seq()
            .add(nn::linear(&p / "1", 4, 2, Default::default()));
        let model2 = nn::seq()
            .add(nn::linear(&p / "1", 4, 2, Default::default()));
        let opt1 = nn::Adam::default().build(&vs, 1e-4).unwrap();
        let opt2 = nn::Adam::default().build(&vs, 1e-4).unwrap();

        Self {
            vs,
            model1,
            model2,
            opt1,
            opt2,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.model1.forward(x)
    }

    pub fn backward_step(&mut self, loss: &Tensor) {
        self.opt1.backward_step(loss);
    }

    pub fn backward(&mut self, loss: &Tensor) {
        self.opt1.zero_grad();
        loss.backward();
    }

    pub fn copy_params(&mut self, src: &nn::VarStore) {
        let src = src.variables();
        let mut dest = self.vs.variables();

        tch::no_grad(|| {
            for name in src.keys() {
                let src = src.get(name).unwrap();
                let dest = dest.get_mut(name).unwrap();
                dest.copy_(&src);
            }
        });
    }

    pub fn copy_grads(&mut self, src: &nn::VarStore) {
        let src = src.variables();
        let mut dest = self.vs.variables();
        self.opt1.zero_grad();

        tch::no_grad(|| {
            for name in src.keys() {
                println!("{:?}", name);
                let src = src.get(name).unwrap();
                println!("src");
                println!("{:?}", src.grad().size());
                let dest = dest.get_mut(name).unwrap();
                println!("dest");
                println!("{:?}", dest.grad().size());
                let mut g = dest.grad();
                g += 1; // it causes panic
                g *= src.grad();
                println!("copy");
            }
        });
    }
}

fn create_data() -> (Tensor, Tensor, Tensor) {
    let w = Tensor::try_from(vec![1., 2., 3., 4., 5., 6., 7., 8.])
        .unwrap()
        .reshape(&[4, 2])
        .internal_cast_float(false);
    let x = Tensor::zeros(&[100, 4], tch::kind::FLOAT_CPU).normal_(0., 1.);
    let n = Tensor::zeros(&[100, 2], tch::kind::FLOAT_CPU).normal_(0., 1.);
    let y = x.matmul(&w) + 0.1 * n;

    (w, x, y)
}

fn test1() {
    tch::manual_seed(1);

    let (_w, xs, ys) = create_data();
    let mut model = LinearModel::new();

    for i in 0..100000 {
        let pred = model.forward(&xs);
        let loss = pred.mse_loss(&ys, tch::Reduction::Mean);
        model.backward_step(&loss);

        if (i + 1) % 10000 == 0 {
            println!("{:?}", ((i + 1), loss));
        }
    }
}

fn test2() {
    tch::manual_seed(1);

    let (_w, xs, ys) = create_data();
    let mut model1 = LinearModel::new();
    let mut model2 = LinearModel::new();

    model2.copy_params(&model1.vs);

    for i in 0..100000 {
        let pred = model2.forward(&xs);
        let loss = pred.mse_loss(&ys, tch::Reduction::Mean);
        model2.backward(&loss);
        model1.copy_grads(&model2.vs);
        panic!();
        model1.opt1.step();
        model2.copy_params(&model1.vs);

        if (i + 1) % 10000 == 0 {
            println!("{:?}", ((i + 1), loss));
        }
    }
}

fn main() {
    test1();
    // test2();
}
