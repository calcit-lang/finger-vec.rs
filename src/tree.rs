//! Internal relaxed B-tree of chunks.
//!
//! Every leaf is a chunk of `1..=CHUNK` elements and every leaf sits at the
//! same depth. Branches keep cumulative sizes, so nodes may hold fewer than
//! `BRANCH` children ("relaxed" nodes, as in RRB vectors). Concatenation joins
//! two trees along their seam and merges or redistributes the seam nodes, so
//! repeated concatenation keeps the height logarithmic.

use std::sync::Arc;

/// Maximum number of elements in one leaf chunk.
pub const CHUNK: usize = 32;
/// Maximum number of children in one branch.
pub const BRANCH: usize = 32;

#[derive(Debug)]
pub struct Branch<T> {
  /// Height of this branch. Leaves have height 0, so branches start at 1.
  pub height: u8,
  /// Cumulative element counts: `ends[i]` is the number of elements in
  /// `children[0..=i]`.
  pub ends: Vec<usize>,
  pub children: Vec<Tree<T>>,
}

#[derive(Debug)]
pub enum Tree<T> {
  Leaf(Arc<Vec<T>>),
  Branch(Arc<Branch<T>>),
}

impl<T> Clone for Tree<T> {
  fn clone(&self) -> Self {
    match self {
      Tree::Leaf(v) => Tree::Leaf(v.clone()),
      Tree::Branch(b) => Tree::Branch(b.clone()),
    }
  }
}

impl<T> Branch<T> {
  pub fn len(&self) -> usize {
    *self.ends.last().unwrap_or(&0)
  }

  /// Index of the child containing element `idx`, plus the offset of that child.
  #[inline]
  pub fn locate(&self, idx: usize) -> (usize, usize) {
    let i = self.ends.partition_point(|&end| end <= idx);
    let offset = if i == 0 { 0 } else { self.ends[i - 1] };
    (i, offset)
  }
}

impl<T: Clone> Tree<T> {
  #[inline]
  pub fn len(&self) -> usize {
    match self {
      Tree::Leaf(v) => v.len(),
      Tree::Branch(b) => b.len(),
    }
  }

  #[inline]
  pub fn height(&self) -> u8 {
    match self {
      Tree::Leaf(_) => 0,
      Tree::Branch(b) => b.height,
    }
  }

  /// Build a branch from children of equal height `height - 1`.
  pub fn branch(height: u8, children: Vec<Tree<T>>) -> Tree<T> {
    let mut ends = Vec::with_capacity(children.len());
    let mut acc = 0;
    for c in &children {
      debug_assert_eq!(c.height() + 1, height);
      acc += c.len();
      ends.push(acc);
    }
    Tree::Branch(Arc::new(Branch { height, ends, children }))
  }

  /// Collapse roots that only have one child.
  pub fn normalize_root(mut self) -> Tree<T> {
    loop {
      match &self {
        Tree::Branch(b) if b.children.len() == 1 => {
          let child = b.children[0].clone();
          self = child;
        }
        _ => return self,
      }
    }
  }

  pub fn get(&self, mut idx: usize) -> &T {
    let mut node = self;
    loop {
      match node {
        Tree::Leaf(v) => return &v[idx],
        Tree::Branch(b) => {
          let (i, offset) = b.locate(idx);
          idx -= offset;
          node = &b.children[i];
        }
      }
    }
  }

  pub fn first(&self) -> &T {
    let mut node = self;
    loop {
      match node {
        Tree::Leaf(v) => return &v[0],
        Tree::Branch(b) => node = &b.children[0],
      }
    }
  }

  pub fn last(&self) -> &T {
    let mut node = self;
    loop {
      match node {
        Tree::Leaf(v) => return &v[v.len() - 1],
        Tree::Branch(b) => node = &b.children[b.children.len() - 1],
      }
    }
  }

  pub fn assoc(&self, idx: usize, item: T) -> Tree<T> {
    match self {
      Tree::Leaf(v) => {
        let mut next: Vec<T> = (**v).clone();
        next[idx] = item;
        Tree::Leaf(Arc::new(next))
      }
      Tree::Branch(b) => {
        let (i, offset) = b.locate(idx);
        let mut children = b.children.clone();
        children[i] = b.children[i].assoc(idx - offset, item);
        Tree::Branch(Arc::new(Branch {
          height: b.height,
          ends: b.ends.clone(),
          children,
        }))
      }
    }
  }

  /// Split into `[0, idx)` and `[idx, len)`. Either side may be absent.
  pub fn split(&self, idx: usize) -> (Option<Tree<T>>, Option<Tree<T>>) {
    if idx == 0 {
      return (None, Some(self.clone()));
    }
    if idx >= self.len() {
      return (Some(self.clone()), None);
    }
    match self {
      Tree::Leaf(v) => (
        Some(Tree::Leaf(Arc::new(v[..idx].to_vec()))),
        Some(Tree::Leaf(Arc::new(v[idx..].to_vec()))),
      ),
      Tree::Branch(b) => {
        let (i, offset) = b.locate(idx);
        let (cl, cr) = b.children[i].split(idx - offset);
        let mut left: Vec<Tree<T>> = Vec::with_capacity(i + 1);
        left.extend(b.children[..i].iter().cloned());
        if let Some(c) = cl {
          left.push(c);
        }
        let mut right: Vec<Tree<T>> = Vec::with_capacity(b.children.len() - i);
        if let Some(c) = cr {
          right.push(c);
        }
        right.extend(b.children[i + 1..].iter().cloned());
        let l = if left.is_empty() {
          None
        } else {
          Some(Tree::branch(b.height, left))
        };
        let r = if right.is_empty() {
          None
        } else {
          Some(Tree::branch(b.height, right))
        };
        (l, r)
      }
    }
  }

  /// Concatenate two trees, keeping all leaves at the same depth.
  pub fn concat(a: &Tree<T>, b: &Tree<T>) -> Tree<T> {
    let mut parts = join(a, b);
    if parts.len() == 1 {
      parts.pop().unwrap().normalize_root()
    } else {
      let h = parts[0].height() + 1;
      Tree::branch(h, parts)
    }
  }

  pub fn traverse_leaves<'a>(&'a self, f: &mut dyn FnMut(&'a [T])) {
    match self {
      Tree::Leaf(v) => f(v),
      Tree::Branch(b) => {
        for c in &b.children {
          c.traverse_leaves(f);
        }
      }
    }
  }

  /// Visit leaves, stopping at the first error.
  pub fn try_leaves<'a, S>(&'a self, f: &mut dyn FnMut(&'a [T]) -> Result<(), S>) -> Result<(), S> {
    match self {
      Tree::Leaf(v) => f(v),
      Tree::Branch(b) => {
        for c in &b.children {
          c.try_leaves(f)?;
        }
        Ok(())
      }
    }
  }

  pub fn map<V>(&self, f: &mut dyn FnMut(&T) -> V) -> Tree<V> {
    match self {
      Tree::Leaf(v) => Tree::Leaf(Arc::new(v.iter().map(&mut *f).collect())),
      Tree::Branch(b) => Tree::Branch(Arc::new(Branch {
        height: b.height,
        ends: b.ends.clone(),
        children: b.children.iter().map(|c| c.map(f)).collect(),
      })),
    }
  }

  pub fn check(&self) -> Result<(), String> {
    match self {
      Tree::Leaf(v) => {
        if v.is_empty() || v.len() > CHUNK {
          Err(format!("leaf size {} out of range", v.len()))
        } else {
          Ok(())
        }
      }
      Tree::Branch(b) => {
        if b.children.is_empty() || b.children.len() > BRANCH {
          return Err(format!("branch with {} children", b.children.len()));
        }
        if b.ends.len() != b.children.len() {
          return Err("ends and children length differ".to_string());
        }
        let mut acc = 0;
        for (i, c) in b.children.iter().enumerate() {
          if c.height() + 1 != b.height {
            return Err(format!("child height {} under branch height {}", c.height(), b.height));
          }
          acc += c.len();
          if b.ends[i] != acc {
            return Err(format!("bad cumulative size at {i}: {} != {acc}", b.ends[i]));
          }
          c.check()?;
        }
        Ok(())
      }
    }
  }

  pub fn format_inline(&self, out: &mut String)
  where
    T: std::fmt::Display,
  {
    match self {
      Tree::Leaf(v) => {
        out.push('[');
        for (i, x) in v.iter().enumerate() {
          if i > 0 {
            out.push(' ');
          }
          out.push_str(&x.to_string());
        }
        out.push(']');
      }
      Tree::Branch(b) => {
        out.push('(');
        for (i, c) in b.children.iter().enumerate() {
          if i > 0 {
            out.push(' ');
          }
          c.format_inline(out);
        }
        out.push(')');
      }
    }
  }
}

/// Build a balanced tree from full chunks.
pub fn build_from_chunks<T: Clone>(mut level: Vec<Tree<T>>) -> Option<Tree<T>> {
  if level.is_empty() {
    return None;
  }
  let mut height = 0u8;
  while level.len() > 1 {
    height += 1;
    let mut next = Vec::with_capacity(level.len().div_ceil(BRANCH));
    let count = level.len();
    // spread children evenly so no node is left nearly empty
    let groups = count.div_ceil(BRANCH);
    let base = count / groups;
    let extra = count % groups;
    let mut iter = level.into_iter();
    for g in 0..groups {
      let size = base + usize::from(g < extra);
      let children: Vec<Tree<T>> = iter.by_ref().take(size).collect();
      next.push(Tree::branch(height, children));
    }
    level = next;
  }
  level.pop()
}

/// Split owned values into evenly sized chunks of at most `CHUNK` elements.
pub fn chunk_values<T>(xs: Vec<T>) -> Vec<Tree<T>> {
  let n = xs.len();
  if n == 0 {
    return vec![];
  }
  let groups = n.div_ceil(CHUNK);
  let base = n / groups;
  let extra = n % groups;
  let mut out = Vec::with_capacity(groups);
  let mut iter = xs.into_iter();
  for g in 0..groups {
    let size = base + usize::from(g < extra);
    let chunk: Vec<T> = iter.by_ref().take(size).collect();
    out.push(Tree::Leaf(Arc::new(chunk)));
  }
  out
}

/// Join two trees and return one or two trees of height `max(ha, hb)`.
fn join<T: Clone>(a: &Tree<T>, b: &Tree<T>) -> Vec<Tree<T>> {
  let ha = a.height();
  let hb = b.height();
  if ha > hb {
    let Tree::Branch(ab) = a else { unreachable!() };
    let last = ab.children.len() - 1;
    let mid = join(&ab.children[last], b);
    let mut children: Vec<Tree<T>> = Vec::with_capacity(ab.children.len() + 1);
    children.extend(ab.children[..last].iter().cloned());
    let seam = children.len();
    let mid_len = mid.len();
    children.extend(mid);
    merge_seam(&mut children, seam.saturating_sub(1), seam + mid_len);
    pack(ha, children)
  } else if ha < hb {
    let Tree::Branch(bb) = b else { unreachable!() };
    let mid = join(a, &bb.children[0]);
    let mid_len = mid.len();
    let mut children: Vec<Tree<T>> = Vec::with_capacity(bb.children.len() + 1);
    children.extend(mid);
    children.extend(bb.children[1..].iter().cloned());
    merge_seam(&mut children, 0, mid_len + 1);
    pack(hb, children)
  } else {
    match (a, b) {
      (Tree::Leaf(x), Tree::Leaf(y)) => join_leaves(x, y),
      (Tree::Branch(ab), Tree::Branch(bb)) => {
        let la = ab.children.len() - 1;
        let mid = join(&ab.children[la], &bb.children[0]);
        let mut children: Vec<Tree<T>> = Vec::with_capacity(ab.children.len() + bb.children.len());
        children.extend(ab.children[..la].iter().cloned());
        let seam = children.len();
        let mid_len = mid.len();
        children.extend(mid);
        children.extend(bb.children[1..].iter().cloned());
        merge_seam(&mut children, seam.saturating_sub(1), seam + mid_len + 1);
        pack(ha, children)
      }
      _ => unreachable!("equal heights imply same node kind"),
    }
  }
}

/// Merge seam leaves when they fit in one chunk, or rebalance a small one.
fn join_leaves<T: Clone>(x: &Arc<Vec<T>>, y: &Arc<Vec<T>>) -> Vec<Tree<T>> {
  let total = x.len() + y.len();
  if total <= CHUNK {
    let mut v = Vec::with_capacity(total);
    v.extend_from_slice(x);
    v.extend_from_slice(y);
    vec![Tree::Leaf(Arc::new(v))]
  } else if x.len() < CHUNK / 2 || y.len() < CHUNK / 2 {
    let left_size = total / 2;
    let mut all: Vec<T> = Vec::with_capacity(total);
    all.extend_from_slice(x);
    all.extend_from_slice(y);
    let right = all.split_off(left_size);
    vec![Tree::Leaf(Arc::new(all)), Tree::Leaf(Arc::new(right))]
  } else {
    vec![Tree::Leaf(x.clone()), Tree::Leaf(y.clone())]
  }
}

/// Merge adjacent nodes in `children[from..to]` whose contents fit in one
/// node. This keeps seams created by concatenation from accumulating
/// nearly empty nodes.
fn merge_seam<T: Clone>(children: &mut Vec<Tree<T>>, from: usize, to: usize) {
  let mut i = from;
  let mut end = to.min(children.len());
  while i + 1 < end {
    let merged = match (&children[i], &children[i + 1]) {
      (Tree::Leaf(x), Tree::Leaf(y)) if x.len() + y.len() <= CHUNK => {
        let mut v = Vec::with_capacity(x.len() + y.len());
        v.extend_from_slice(x);
        v.extend_from_slice(y);
        Some(Tree::Leaf(Arc::new(v)))
      }
      (Tree::Branch(x), Tree::Branch(y)) if x.children.len() + y.children.len() <= BRANCH => {
        let mut cs = Vec::with_capacity(x.children.len() + y.children.len());
        cs.extend(x.children.iter().cloned());
        cs.extend(y.children.iter().cloned());
        Some(Tree::branch(x.height, cs))
      }
      _ => None,
    };
    match merged {
      Some(node) => {
        children[i] = node;
        children.remove(i + 1);
        end -= 1;
      }
      None => i += 1,
    }
  }
}

/// Wrap children of height `height - 1` into one or two branches of `height`.
fn pack<T: Clone>(height: u8, mut children: Vec<Tree<T>>) -> Vec<Tree<T>> {
  if children.len() <= BRANCH {
    vec![Tree::branch(height, children)]
  } else {
    let right = children.split_off(children.len() / 2);
    vec![Tree::branch(height, children), Tree::branch(height, right)]
  }
}
