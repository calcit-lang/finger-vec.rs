//! Quick timing of common operations. Run with `cargo run --release --example bench`.

use finger_vec::FingerVec;
use std::time::Instant;

fn time(name: &str, f: impl FnOnce()) {
  let start = Instant::now();
  f();
  println!("{name:<36} {:>9.2} ms", start.elapsed().as_secs_f64() * 1e3);
}

fn main() {
  let n = 100_000usize;
  let mut sink = 0usize;
  let mut v = FingerVec::new();
  time("push_right 100k", || {
    for i in 0..n {
      v = v.push_right(i);
    }
  });
  let mut p = FingerVec::new();
  time("push_left 100k", || {
    for i in 0..n {
      p = p.push_left(i);
    }
  });
  let source: Vec<usize> = (0..n).collect();
  let mut b = FingerVec::new();
  time("from Vec 100k", || b = FingerVec::from(source.clone()));
  let mut x = 1u64;
  let idx: Vec<usize> = (0..n)
    .map(|_| {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      (x as usize) % n
    })
    .collect();
  time("get 100k random", || {
    for &i in &idx {
      sink += *b.get(i).unwrap();
    }
  });
  time("iterate 100k", || {
    for y in v.iter() {
      sink += *y;
    }
  });
  let mut r = v.clone();
  time("drop_left until empty 100k", || {
    while !r.is_empty() {
      r = r.drop_left();
    }
  });
  let mut u = b.clone();
  time("assoc 100k random", || {
    for &i in &idx {
      u = u.assoc(i, i).unwrap();
    }
  });
  let mut c = FingerVec::new();
  time("concat acc+[x] 20k", || {
    for i in 0..20_000 {
      c = c.concat_with(&FingerVec::from(vec![i]));
    }
  });
  let mut m = b.clone();
  time("insert middle 20k", || {
    for i in 0..20_000 {
      let l = m.len();
      m = m.insert(l / 2, i, false).unwrap();
    }
  });
  println!("depths: pushed {} concat {} inserted {}", v.depth(), c.depth(), m.depth());
  println!("sink {sink}");
}
