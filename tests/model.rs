//! Randomized model tests against `Vec<T>`.

use finger_vec::FingerVec;

struct Rng(u64);
impl Rng {
  fn next(&mut self) -> u64 {
    self.0 ^= self.0 << 13;
    self.0 ^= self.0 >> 7;
    self.0 ^= self.0 << 17;
    self.0
  }
  fn below(&mut self, n: usize) -> usize {
    if n == 0 { 0 } else { (self.next() % n as u64) as usize }
  }
}

fn check(v: &FingerVec<u32>, m: &[u32]) {
  v.check_structure()
    .unwrap_or_else(|e| panic!("structure: {e}\n{}", v.format_inline()));
  assert_eq!(v.len(), m.len());
  assert_eq!(v.is_empty(), m.is_empty());
  assert_eq!(v.to_vec(), m);
  assert_eq!(v.iter().copied().collect::<Vec<_>>(), m);
  assert_eq!(v.iter().len(), m.len());
  assert_eq!(v.first(), m.first());
  assert_eq!(v.last(), m.last());
  for (i, x) in m.iter().enumerate().step_by(7) {
    assert_eq!(v.get(i), Some(x));
  }
  assert_eq!(v.get(m.len()), None);
  let mut walked = vec![];
  v.traverse(&mut |x| walked.push(*x));
  assert_eq!(walked, m);
}

fn random_vec(rng: &mut Rng, counter: &mut u32) -> (FingerVec<u32>, Vec<u32>) {
  let n = match rng.below(4) {
    0 => rng.below(4),
    1 => rng.below(40),
    2 => rng.below(300),
    _ => rng.below(3000),
  };
  let mut m = Vec::with_capacity(n);
  for _ in 0..n {
    *counter += 1;
    m.push(*counter);
  }
  let v = if rng.below(2) == 0 {
    FingerVec::from(m.clone())
  } else {
    let mut v = FingerVec::new();
    if rng.below(2) == 0 {
      for x in &m {
        v = v.push_right(*x);
      }
    } else {
      for x in m.iter().rev() {
        v = v.push_left(*x);
      }
    }
    v
  };
  (v, m)
}

fn run(seed: u64, steps: usize) {
  let mut rng = Rng(seed);
  let mut counter = 0u32;
  let (mut v, mut m) = random_vec(&mut rng, &mut counter);
  let mut snapshots: Vec<(FingerVec<u32>, Vec<u32>)> = vec![];
  for step in 0..steps {
    counter += 1;
    let x = counter;
    match rng.below(14) {
      0 | 1 => {
        v = v.push_right(x);
        m.push(x);
      }
      2 | 3 => {
        v = v.push_left(x);
        m.insert(0, x);
      }
      4 => {
        v = v.drop_left();
        if !m.is_empty() {
          m.remove(0);
        }
      }
      5 => {
        v = v.drop_right();
        m.pop();
      }
      6 => {
        if !m.is_empty() {
          let i = rng.below(m.len());
          v = v.assoc(i, x).unwrap();
          m[i] = x;
        } else {
          assert!(v.assoc(0, x).is_err());
        }
      }
      7 => {
        if !m.is_empty() {
          let i = rng.below(m.len());
          let after = rng.below(2) == 0;
          v = v.insert(i, x, after).unwrap();
          m.insert(if after { i + 1 } else { i }, x);
        }
      }
      8 => {
        if !m.is_empty() {
          let i = rng.below(m.len());
          v = v.dissoc(i).unwrap();
          m.remove(i);
        }
      }
      9 => {
        let a = rng.below(m.len() + 1);
        let b = a + rng.below(m.len() - a + 1);
        v = v.slice(a, b).unwrap();
        m = m[a..b].to_vec();
      }
      10 | 11 => {
        let (w, wm) = random_vec(&mut rng, &mut counter);
        if rng.below(2) == 0 {
          v = v.concat_with(&w);
          m.extend(wm);
        } else {
          v = w.concat_with(&v);
          let mut z = wm;
          z.extend(m);
          m = z;
        }
      }
      12 => {
        let i = rng.below(m.len() + 1);
        let (l, r) = v.split(i);
        check(&l, &m[..i]);
        check(&r, &m[i..]);
        v = if rng.below(2) == 0 {
          l.concat_with(&r)
        } else {
          FingerVec::concat(&[l, r])
        };
      }
      _ => {
        v = v.reverse();
        m.reverse();
      }
    }
    check(&v, &m);
    if step % 50 == 0 {
      snapshots.push((v.clone(), m.clone()));
    }
    // keep sizes bounded so the test stays fast
    if m.len() > 20_000 {
      v = v.slice(0, 5_000).unwrap();
      m.truncate(5_000);
    }
  }
  // persistence: earlier versions are unchanged
  for (sv, sm) in &snapshots {
    check(sv, sm);
  }
}

#[test]
fn model_many_seeds() {
  for seed in 1..=40u64 {
    run(seed * 7919, 400);
  }
}

#[test]
fn model_long_run() {
  run(123_456_789, 4000);
}

#[test]
fn reverse_iteration_and_lookup() {
  let m: Vec<u32> = (0..1000).collect();
  let v = FingerVec::from(m.clone());
  let rev: Vec<u32> = v.iter().rev().copied().collect();
  let mut expected = m.clone();
  expected.reverse();
  assert_eq!(rev, expected);
  assert_eq!(v.index_of(&500), Some(500));
  assert_eq!(v.last_index_of(&3), Some(3));
  assert_eq!(v.find_index(|x| *x > 998), Some(999));
  let mut it = v.iter();
  assert_eq!(it.next(), Some(&0));
  assert_eq!(it.next_back(), Some(&999));
  assert_eq!(it.len(), 998);
}

#[test]
fn errors_match_previous_semantics() {
  let v = FingerVec::from(vec![1, 2, 3]);
  assert!(v.slice(2, 4).is_err());
  assert!(v.slice(3, 2).is_err());
  assert_eq!(v.slice(1, 1).unwrap().len(), 0);
  assert!(v.skip(4).is_err());
  assert!(v.take(4).is_err());
  assert!(v.assoc(3, 0).is_err());
  assert!(v.dissoc(3).is_err());
  assert!(v.insert(3, 0, false).is_err());
  let e = FingerVec::<u32>::new();
  assert!(e.rest().is_err());
  assert!(e.butlast().is_err());
  assert_eq!(e.insert(0, 9, false).unwrap().to_vec(), vec![9]);
  assert!(e.insert(1, 9, false).is_err());
  assert_eq!(e.drop_left().len(), 0);
}

#[test]
fn equality_order_and_hash_follow_vec() {
  use std::collections::hash_map::DefaultHasher;
  use std::hash::{Hash, Hasher};
  let a: FingerVec<u32> = (0..100).collect();
  let mut b = FingerVec::new();
  for i in (0..100).rev() {
    b = b.push_left(i);
  }
  assert_eq!(a, b);
  let h = |x: &FingerVec<u32>| {
    let mut s = DefaultHasher::new();
    x.hash(&mut s);
    s.finish()
  };
  assert_eq!(h(&a), h(&b));
  let c = FingerVec::from(vec![1u32, 2]);
  let d = FingerVec::from(vec![1u32, 2, 0]);
  let e = FingerVec::from(vec![2u32]);
  assert!(c < d);
  assert!(d < e);
  assert_eq!(c.cmp(&d), vec![1u32, 2].cmp(&vec![1, 2, 0]));
}
