use tch::{nn::ModuleT, Tensor, kind::Kind::Float};

fn main() {
    // X: data
    let data = Tensor::of_slice((0..24).collect::<Vec<_>>().as_slice())
        .reshape(&[2, 3, 4])
        .internal_cast_float(false);
    data.print();

    // W: Weight
    let w = Tensor::of_slice((0..16).collect::<Vec<_>>().as_slice())
        .reshape(&[4, 4])
        .internal_cast_float(false);
    w.print();

    // A: Neighbor
    let a = Tensor::of_slice(&[1, 0, 1, 0, 1, 1, 0, 1, 0])
        .reshape(&[3, 3])
        .internal_cast_float(false);
    a.print();

    // XW
    let data = data.matmul(&w);
    data.print();

    // AXW
    let data = a.matmul(&data);
    data.print();

    // D
    let d = a.sum1(&[1], false, Float).sqrt().diag(0);
    d.print();
    let d = d.inverse();
    d.print();

    // DAD
    let n = a.size()[0];
    let a = a + Tensor::eye(n, (Float, tch::Device::Cpu));
    a.print();
}
