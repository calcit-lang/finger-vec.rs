//! Height stays logarithmic under patterns that degrade unbalanced trees.

use finger_vec::FingerVec;

fn bound(n: usize) -> usize {
  // generous: log base 4 of chunk count plus slack
  let mut c = n.div_ceil(8).max(1);
  let mut h = 1;
  while c > 1 {
    c = c.div_ceil(8);
    h += 1;
  }
  h + 1
}

#[test]
fn concat_single_items_in_a_loop() {
  let mut right = FingerVec::new();
  let mut left = FingerVec::new();
  for i in 0..20_000u32 {
    right = right.concat_with(&FingerVec::from(vec![i]));
    left = FingerVec::from(vec![i]).concat_with(&left);
  }
  for v in [&right, &left] {
    v.check_structure().unwrap();
    assert!(v.depth() <= bound(v.len()), "depth {} for {}", v.depth(), v.len());
  }
  assert_eq!(right.get(12_345), Some(&12_345));
  assert_eq!(left.get(0), Some(&19_999));
}

#[test]
fn concat_medium_lists_in_a_loop() {
  let mut acc = FingerVec::new();
  for i in 0..2_000u32 {
    let chunk: FingerVec<u32> = (i * 50..i * 50 + 50).collect();
    acc = if i % 2 == 0 {
      acc.concat_with(&chunk)
    } else {
      FingerVec::concat(&[acc, chunk])
    };
  }
  acc.check_structure().unwrap();
  assert_eq!(acc.len(), 100_000);
  assert!(acc.depth() <= bound(acc.len()), "depth {}", acc.depth());
  for i in (0..100_000).step_by(997) {
    assert_eq!(acc.get(i), Some(&(i as u32)));
  }
}

#[test]
fn insert_in_the_middle() {
  let mut v: FingerVec<u32> = (0..1000).collect();
  for i in 0..20_000u32 {
    let mid = v.len() / 2;
    v = v.insert(mid, i, false).unwrap();
  }
  v.check_structure().unwrap();
  assert_eq!(v.len(), 21_000);
  assert!(v.depth() <= bound(v.len()), "depth {}", v.depth());
}

#[test]
fn queue_usage() {
  let mut q = FingerVec::new();
  for i in 0..100_000u32 {
    q = q.push_right(i);
    if i % 2 == 1 {
      q = q.drop_left();
    }
  }
  q.check_structure().unwrap();
  assert_eq!(q.len(), 50_000);
  assert_eq!(q.first(), Some(&50_000));
}

#[test]
fn rest_loop_drains() {
  let mut v: FingerVec<u32> = (0..10_000).collect();
  let mut n = 0;
  while let Some(x) = v.first().copied() {
    assert_eq!(x, n);
    v = v.rest().unwrap();
    n += 1;
  }
  assert_eq!(n, 10_000);
}
