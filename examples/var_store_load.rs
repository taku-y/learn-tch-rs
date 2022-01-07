
use tch::nn;

fn main() {
    let vs1 = nn::VarStore::new(tch::Device::Cpu);
    let p1 = vs1.root();
    let _seq1 = nn::seq()
        .add(nn::linear(&p1 / "1", 4, 2, Default::default()))
        .add(nn::linear(&p1 / "2", 2, 1, Default::default()));
    vs1.save("./tmp.pth").unwrap();
    
    let mut vs2 = nn::VarStore::new(tch::Device::Cpu);
    let p2 = vs2.root();
    let _seq2 = nn::seq()
        .add(nn::linear(&p2 / "1", 4, 2, Default::default()));
    vs2.load_partial("./tmp.pth").unwrap();
    vs2.load("./tmp.pth").unwrap();
}
