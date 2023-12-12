pub fn isqrt(num: usize) -> usize {
    let mut sqrt = num;
    let mut next_sqrt = (sqrt + num / sqrt) / 2;
    while next_sqrt < sqrt {
        sqrt = next_sqrt;
        next_sqrt = (sqrt + num / sqrt) / 2;
    }
    sqrt
}
pub fn round(x: f64) -> u32 {
    x.round() as u32
}

pub fn clamp(x: f64, mina: f64, maxa: f64, minb: f64, maxb: f64) -> f64 {
    assert!(x >= mina);
    assert!(x <= maxa);
    (x - mina) * (maxb - minb) / (maxa - mina) + minb
}

