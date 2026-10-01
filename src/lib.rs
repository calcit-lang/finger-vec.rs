//! `FingerVec<T>`: a persistent vector for the Calcit runtime.
//!
//! Layout: a front buffer, a relaxed B-tree of chunks in the middle, and a
//! back buffer. Buffers make operations on both ends cheap; the tree keeps
//! indexing, slicing and concatenation logarithmic.
//!
//! - `get`, `assoc`: `O(log₃₂ n)`
//! - `push_left`, `push_right`, `drop_left`, `drop_right`: `O(1)` amortized
//!   with a chunk copy of at most 32 elements
//! - `concat`, `split`, `slice`, `insert`, `dissoc`: `O(log n)`
//! - lists with at most 32 elements live in one buffer
//!
//! All operations return new values and share structure with the input.

mod tree;

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::sync::Arc;

pub use tree::{BRANCH, CHUNK};
use tree::{Tree, build_from_chunks, chunk_values};

/// A slice of a shared chunk: `data[start..end]`.
#[derive(Debug)]
struct Buf<T> {
  data: Option<Arc<Vec<T>>>,
  start: usize,
  end: usize,
}

impl<T> Clone for Buf<T> {
  fn clone(&self) -> Self {
    Buf {
      data: self.data.clone(),
      start: self.start,
      end: self.end,
    }
  }
}

impl<T: Clone> Buf<T> {
  const fn empty() -> Self {
    Buf {
      data: None,
      start: 0,
      end: 0,
    }
  }

  fn from_vec(v: Vec<T>) -> Self {
    if v.is_empty() {
      return Self::empty();
    }
    let end = v.len();
    Buf {
      data: Some(Arc::new(v)),
      start: 0,
      end,
    }
  }

  fn from_arc(v: Arc<Vec<T>>) -> Self {
    let end = v.len();
    Buf {
      data: Some(v),
      start: 0,
      end,
    }
  }

  #[inline]
  fn len(&self) -> usize {
    self.end - self.start
  }

  #[inline]
  fn as_slice(&self) -> &[T] {
    match &self.data {
      Some(v) => &v[self.start..self.end],
      None => &[],
    }
  }

  /// Take the buffer as an exact chunk, reusing the allocation when possible.
  fn into_chunk(self) -> Option<Arc<Vec<T>>> {
    match self.data {
      None => None,
      Some(_) if self.start == self.end => None,
      Some(v) if self.start == 0 && self.end == v.len() => Some(v),
      Some(v) => Some(Arc::new(v[self.start..self.end].to_vec())),
    }
  }

  /// Make the buffer uniquely owned and exactly sized, then return the vector.
  fn make_mut(&mut self) -> &mut Vec<T> {
    let (start, end) = (self.start, self.end);
    let data = self.data.get_or_insert_with(|| Arc::new(Vec::with_capacity(4)));
    if let Some(v) = Arc::get_mut(data) {
      v.truncate(end);
      if start > 0 {
        v.drain(..start);
      }
    } else {
      let mut v = Vec::with_capacity((end - start + 1).next_power_of_two().min(tree::CHUNK));
      v.extend_from_slice(&data[start..end]);
      *data = Arc::new(v);
    }
    self.start = 0;
    let v = Arc::get_mut(data).expect("buffer is unique");
    self.end = v.len();
    v
  }

  fn push_back(&mut self, x: T) {
    let v = self.make_mut();
    v.push(x);
    self.end = v.len();
  }

  fn push_front(&mut self, x: T) {
    let v = self.make_mut();
    v.insert(0, x);
    self.end = v.len();
  }

  fn set(&mut self, idx: usize, x: T) {
    let v = self.make_mut();
    v[idx] = x;
  }
}

/// Persistent vector with cheap operations at both ends.
#[derive(Debug)]
pub struct FingerVec<T> {
  front: Buf<T>,
  tree: Option<Tree<T>>,
  back: Buf<T>,
}

impl<T> Clone for FingerVec<T> {
  fn clone(&self) -> Self {
    FingerVec {
      front: self.front.clone(),
      tree: self.tree.clone(),
      back: self.back.clone(),
    }
  }
}

impl<T: Clone> Default for FingerVec<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: Clone> FingerVec<T> {
  /// An empty vector. Does not allocate.
  pub const fn new() -> Self {
    FingerVec {
      front: Buf::empty(),
      tree: None,
      back: Buf::empty(),
    }
  }

  fn tree_len(&self) -> usize {
    self.tree.as_ref().map_or(0, |t| t.len())
  }

  pub fn len(&self) -> usize {
    self.front.len() + self.tree_len() + self.back.len()
  }

  pub fn is_empty(&self) -> bool {
    self.front.len() == 0 && self.tree.is_none() && self.back.len() == 0
  }

  /// Height of the middle tree, `0` when there is no tree. Useful in tests.
  pub fn depth(&self) -> usize {
    self.tree.as_ref().map_or(0, |t| t.height() as usize + 1)
  }

  pub fn get(&self, idx: usize) -> Option<&T> {
    let f = self.front.len();
    if idx < f {
      return Some(&self.front.as_slice()[idx]);
    }
    let idx = idx - f;
    let t = self.tree_len();
    if idx < t {
      return self.tree.as_ref().map(|tree| tree.get(idx));
    }
    self.back.as_slice().get(idx - t)
  }

  pub fn first(&self) -> Option<&T> {
    if let Some(x) = self.front.as_slice().first() {
      return Some(x);
    }
    if let Some(t) = &self.tree {
      return Some(t.first());
    }
    self.back.as_slice().first()
  }

  pub fn last(&self) -> Option<&T> {
    if let Some(x) = self.back.as_slice().last() {
      return Some(x);
    }
    if let Some(t) = &self.tree {
      return Some(t.last());
    }
    self.front.as_slice().last()
  }

  // ---------- mutation on owned values (used by builders and the persistent API) ----------

  /// Append in place. Cheap when this value is not shared.
  pub fn push_back_mut(&mut self, x: T) {
    if self.back.len() >= CHUNK {
      let chunk = std::mem::replace(&mut self.back, Buf::empty()).into_chunk();
      if let Some(chunk) = chunk {
        let leaf = Tree::Leaf(chunk);
        self.tree = Some(match self.tree.take() {
          None => leaf,
          Some(t) => Tree::concat(&t, &leaf),
        });
      }
    }
    self.back.push_back(x);
  }

  /// Prepend in place. Cheap when this value is not shared.
  pub fn push_front_mut(&mut self, x: T) {
    if self.front.len() >= CHUNK {
      let chunk = std::mem::replace(&mut self.front, Buf::empty()).into_chunk();
      if let Some(chunk) = chunk {
        let leaf = Tree::Leaf(chunk);
        self.tree = Some(match self.tree.take() {
          None => leaf,
          Some(t) => Tree::concat(&leaf, &t),
        });
      }
    }
    // a small list lives in `back`; keep it there so it stays one chunk
    if self.front.len() == 0 && self.tree.is_none() && self.back.len() < CHUNK {
      self.back.push_front(x);
    } else {
      self.front.push_front(x);
    }
  }

  /// Remove the first element in place. No-op on an empty vector.
  pub fn pop_front_mut(&mut self) -> Option<T> {
    if self.front.len() == 0 {
      if let Some(t) = self.tree.take() {
        let first_leaf_len = leftmost_leaf_len(&t);
        let (head, rest) = t.split(first_leaf_len);
        self.tree = rest.map(|r| r.normalize_root());
        if let Some(Tree::Leaf(v)) = head.map(|h| h.normalize_root()) {
          self.front = Buf::from_arc(v);
        }
      } else if self.back.len() > 0 {
        let x = self.back.as_slice()[0].clone();
        self.back.start += 1;
        if self.back.len() == 0 {
          self.back = Buf::empty();
        }
        return Some(x);
      } else {
        return None;
      }
    }
    let x = self.front.as_slice()[0].clone();
    self.front.start += 1;
    if self.front.len() == 0 {
      self.front = Buf::empty();
    }
    Some(x)
  }

  /// Remove the last element in place. No-op on an empty vector.
  pub fn pop_back_mut(&mut self) -> Option<T> {
    if self.back.len() == 0 {
      if let Some(t) = self.tree.take() {
        let len = t.len();
        let last_leaf_len = rightmost_leaf_len(&t);
        let (rest, tail) = t.split(len - last_leaf_len);
        self.tree = rest.map(|r| r.normalize_root());
        if let Some(Tree::Leaf(v)) = tail.map(|h| h.normalize_root()) {
          self.back = Buf::from_arc(v);
        }
      } else if self.front.len() > 0 {
        let x = self.front.as_slice()[self.front.len() - 1].clone();
        self.front.end -= 1;
        if self.front.len() == 0 {
          self.front = Buf::empty();
        }
        return Some(x);
      } else {
        return None;
      }
    }
    let x = self.back.as_slice()[self.back.len() - 1].clone();
    self.back.end -= 1;
    if self.back.len() == 0 {
      self.back = Buf::empty();
    }
    Some(x)
  }

  // ---------- persistent API ----------

  pub fn push_right(&self, x: T) -> Self {
    let mut next = self.clone();
    next.push_back_mut(x);
    next
  }

  pub fn push_left(&self, x: T) -> Self {
    let mut next = self.clone();
    next.push_front_mut(x);
    next
  }

  /// Alias of [`FingerVec::push_right`].
  pub fn push(&self, x: T) -> Self {
    self.push_right(x)
  }

  /// Alias of [`FingerVec::push_right`].
  pub fn append(&self, x: T) -> Self {
    self.push_right(x)
  }

  /// Alias of [`FingerVec::push_left`].
  pub fn prepend(&self, x: T) -> Self {
    self.push_left(x)
  }

  /// Alias of [`FingerVec::push_left`].
  pub fn unshift(&self, x: T) -> Self {
    self.push_left(x)
  }

  /// Without the first element. Empty stays empty.
  pub fn drop_left(&self) -> Self {
    let mut next = self.clone();
    next.pop_front_mut();
    next
  }

  /// Without the last element. Empty stays empty.
  pub fn drop_right(&self) -> Self {
    let mut next = self.clone();
    next.pop_back_mut();
    next
  }

  pub fn rest(&self) -> Result<Self, String> {
    if self.is_empty() {
      Err(String::from("calling rest on empty"))
    } else {
      Ok(self.drop_left())
    }
  }

  pub fn butlast(&self) -> Result<Self, String> {
    if self.is_empty() {
      Err(String::from("calling butlast on empty"))
    } else {
      Ok(self.drop_right())
    }
  }

  /// Replace the element at `idx`.
  pub fn assoc(&self, idx: usize, item: T) -> Result<Self, String> {
    let len = self.len();
    if idx >= len {
      return Err(format!("Index too large {idx} for list of size {len}"));
    }
    let mut next = self.clone();
    let f = next.front.len();
    if idx < f {
      next.front.set(idx, item);
      return Ok(next);
    }
    let i = idx - f;
    let t = next.tree_len();
    if i < t {
      next.tree = next.tree.as_ref().map(|tree| tree.assoc(i, item));
    } else {
      next.back.set(i - t, item);
    }
    Ok(next)
  }

  /// Split into `[0, idx)` and `[idx, len)`. An index past the end returns `(self, empty)`.
  pub fn split(&self, idx: usize) -> (Self, Self) {
    let len = self.len();
    if idx == 0 {
      return (Self::new(), self.clone());
    }
    if idx >= len {
      return (self.clone(), Self::new());
    }
    let f = self.front.len();
    if idx <= f {
      let left = Self::from_slice_buf(&self.front.as_slice()[..idx]);
      let mut right = self.clone();
      right.front.start += idx;
      if right.front.len() == 0 {
        right.front = Buf::empty();
      }
      return (left, right);
    }
    let t = self.tree_len();
    if idx <= f + t {
      let (tl, tr) = self.tree.as_ref().expect("tree exists").split(idx - f);
      let left = FingerVec {
        front: self.front.clone(),
        tree: tl.map(|x| x.normalize_root()),
        back: Buf::empty(),
      };
      let right = FingerVec {
        front: Buf::empty(),
        tree: tr.map(|x| x.normalize_root()),
        back: self.back.clone(),
      };
      return (left.rebalanced(), right.rebalanced());
    }
    let k = idx - f - t;
    let mut left = self.clone();
    left.back.end = left.back.start + k;
    if left.back.len() == 0 {
      left.back = Buf::empty();
    }
    let right = Self::from_slice_buf(&self.back.as_slice()[k..]);
    (left, right)
  }

  fn from_slice_buf(xs: &[T]) -> Self {
    if xs.len() <= CHUNK {
      FingerVec {
        front: Buf::empty(),
        tree: None,
        back: Buf::from_vec(xs.to_vec()),
      }
    } else {
      Self::from(xs.to_vec())
    }
  }

  /// Elements in `[start, end)`.
  pub fn slice(&self, start: usize, end: usize) -> Result<Self, String> {
    let len = self.len();
    if end > len {
      return Err(format!("Slice range too large {end} for list of size {len}"));
    }
    if start > end {
      return Err(format!("Invalid slice range {start}..{end}"));
    }
    if start == end {
      return Ok(Self::new());
    }
    let (_, right) = self.split(start);
    let (mid, _) = right.split(end - start);
    Ok(mid)
  }

  /// Without the first `n` elements.
  pub fn skip(&self, n: usize) -> Result<Self, String> {
    let len = self.len();
    if n > len {
      return Err(format!("Skip range too large {n} for list of size {len}"));
    }
    Ok(self.split(n).1)
  }

  /// The first `n` elements.
  pub fn take(&self, n: usize) -> Result<Self, String> {
    let len = self.len();
    if n > len {
      return Err(format!("Take range too large {n} for list of size {len}"));
    }
    Ok(self.split(n).0)
  }

  /// Concatenate two vectors.
  pub fn concat_with(&self, other: &Self) -> Self {
    if other.is_empty() {
      return self.clone();
    }
    if self.is_empty() {
      return other.clone();
    }
    // short inputs: push elements into the buffers
    if other.len() <= CHUNK {
      let mut next = self.clone();
      for x in other.iter() {
        next.push_back_mut(x.clone());
      }
      return next;
    }
    if self.len() <= CHUNK {
      let mut next = other.clone();
      let items: Vec<T> = self.iter().cloned().collect();
      for x in items.into_iter().rev() {
        next.push_front_mut(x);
      }
      return next;
    }
    // left tree = self.tree + self.back; right tree = other.front + other.tree
    let mut left_tree = self.tree.clone();
    if let Some(chunk) = self.back.clone().into_chunk() {
      let leaf = Tree::Leaf(chunk);
      left_tree = Some(match left_tree {
        None => leaf,
        Some(t) => Tree::concat(&t, &leaf),
      });
    }
    let mut right_tree = other.tree.clone();
    if let Some(chunk) = other.front.clone().into_chunk() {
      let leaf = Tree::Leaf(chunk);
      right_tree = Some(match right_tree {
        None => leaf,
        Some(t) => Tree::concat(&leaf, &t),
      });
    }
    let tree = match (left_tree, right_tree) {
      (None, None) => None,
      (Some(a), None) => Some(a),
      (None, Some(b)) => Some(b),
      (Some(a), Some(b)) => Some(Tree::concat(&a, &b)),
    };
    FingerVec {
      front: self.front.clone(),
      tree,
      back: other.back.clone(),
    }
    .rebalanced()
  }

  /// Concatenate many vectors.
  pub fn concat(raw: &[Self]) -> Self {
    let mut acc = Self::new();
    for x in raw {
      acc = acc.concat_with(x);
    }
    acc
  }

  /// Insert `item` so it ends up at `idx` (`after == false`) or `idx + 1` (`after == true`).
  pub fn insert(&self, idx: usize, item: T, after: bool) -> Result<Self, String> {
    let len = self.len();
    if len == 0 {
      return if idx == 0 {
        Ok(self.push_right(item))
      } else {
        Err(String::from("inserting into empty, but index is not 0"))
      };
    }
    if idx >= len {
      return Err(format!("Index too large {idx} for list of size {len}"));
    }
    let pos = if after { idx + 1 } else { idx };
    Ok(self.insert_at(pos, item))
  }

  /// Insert at position `pos` in `0..=len`.
  fn insert_at(&self, pos: usize, item: T) -> Self {
    if pos == 0 {
      return self.push_left(item);
    }
    if pos == self.len() {
      return self.push_right(item);
    }
    let (mut left, right) = self.split(pos);
    left.push_back_mut(item);
    left.concat_with(&right)
  }

  pub fn assoc_before(&self, idx: usize, item: T) -> Result<Self, String> {
    self.insert(idx, item, false)
  }

  pub fn assoc_after(&self, idx: usize, item: T) -> Result<Self, String> {
    self.insert(idx, item, true)
  }

  /// Remove the element at `idx`.
  pub fn dissoc(&self, idx: usize) -> Result<Self, String> {
    let len = self.len();
    if idx >= len {
      return Err(format!("Index too large {idx} for list of size {len}"));
    }
    if idx == 0 {
      return Ok(self.drop_left());
    }
    if idx == len - 1 {
      return Ok(self.drop_right());
    }
    let (left, right) = self.split(idx);
    Ok(left.concat_with(&right.drop_left()))
  }

  pub fn reverse(&self) -> Self {
    let mut items = self.to_vec();
    items.reverse();
    Self::from(items)
  }

  pub fn map<V: Clone>(&self, mut f: impl FnMut(&T) -> V) -> FingerVec<V> {
    let front: Vec<V> = self.front.as_slice().iter().map(&mut f).collect();
    let tree = self.tree.as_ref().map(|t| t.map(&mut f));
    let back: Vec<V> = self.back.as_slice().iter().map(&mut f).collect();
    FingerVec {
      front: Buf::from_vec(front),
      tree,
      back: Buf::from_vec(back),
    }
  }

  pub fn to_vec(&self) -> Vec<T> {
    let mut out = Vec::with_capacity(self.len());
    self.traverse_chunks(&mut |chunk| out.extend_from_slice(chunk));
    out
  }

  /// Visit the elements chunk by chunk.
  pub fn traverse_chunks<'a>(&'a self, f: &mut dyn FnMut(&'a [T])) {
    if self.front.len() > 0 {
      f(self.front.as_slice());
    }
    if let Some(t) = &self.tree {
      t.traverse_leaves(f);
    }
    if self.back.len() > 0 {
      f(self.back.as_slice());
    }
  }

  pub fn traverse(&self, f: &mut dyn FnMut(&T)) {
    self.traverse_chunks(&mut |chunk| {
      for x in chunk {
        f(x);
      }
    });
  }

  /// Visit elements, returning the first error.
  pub fn traverse_result<S>(&self, f: &mut dyn FnMut(&T) -> Result<(), S>) -> Result<(), S> {
    for x in self.front.as_slice() {
      f(x)?;
    }
    if let Some(t) = &self.tree {
      t.try_leaves(&mut |chunk| {
        for x in chunk {
          f(x)?;
        }
        Ok(())
      })?;
    }
    for x in self.back.as_slice() {
      f(x)?;
    }
    Ok(())
  }

  pub fn find_index(&self, f: impl Fn(&T) -> bool) -> Option<usize> {
    self.iter().position(f)
  }

  pub fn iter(&self) -> Iter<'_, T> {
    Iter::new(self)
  }

  /// Check internal invariants. Intended for tests.
  pub fn check_structure(&self) -> Result<(), String> {
    if self.front.start > self.front.end || self.back.start > self.back.end {
      return Err("buffer range is inverted".to_string());
    }
    if self.front.len() > CHUNK || self.back.len() > CHUNK {
      return Err("buffer longer than one chunk".to_string());
    }
    if let Some(t) = &self.tree {
      t.check()?;
      let max = max_height(self.len());
      if (t.height() as usize) > max {
        return Err(format!("tree height {} exceeds {max} for {} items", t.height(), self.len()));
      }
    }
    Ok(())
  }

  /// Rebuild the tree when seams have made it taller than needed.
  fn rebalanced(mut self) -> Self {
    let too_tall = match &self.tree {
      Some(t) => (t.height() as usize) > max_height(self.len()),
      None => false,
    };
    if too_tall {
      let mut items = Vec::with_capacity(self.tree_len());
      if let Some(t) = &self.tree {
        t.traverse_leaves(&mut |c| items.extend_from_slice(c));
      }
      self.tree = build_from_chunks(chunk_values(items));
    }
    self
  }
}

/// Height allowed for a tree holding `len` elements: the height of a tree
/// whose nodes are a quarter full, plus one level of slack.
fn max_height(len: usize) -> usize {
  let mut chunks = len.div_ceil(CHUNK / 4).max(1);
  let mut h = 0;
  while chunks > 1 {
    chunks = chunks.div_ceil(BRANCH / 4);
    h += 1;
  }
  h + 1
}

fn leftmost_leaf_len<T: Clone>(t: &Tree<T>) -> usize {
  let mut node = t;
  loop {
    match node {
      Tree::Leaf(v) => return v.len(),
      Tree::Branch(b) => node = &b.children[0],
    }
  }
}

fn rightmost_leaf_len<T: Clone>(t: &Tree<T>) -> usize {
  let mut node = t;
  loop {
    match node {
      Tree::Leaf(v) => return v.len(),
      Tree::Branch(b) => node = &b.children[b.children.len() - 1],
    }
  }
}

impl<T: Clone + PartialEq> FingerVec<T> {
  pub fn index_of(&self, item: &T) -> Option<usize> {
    self.iter().position(|x| x == item)
  }

  pub fn last_index_of(&self, item: &T) -> Option<usize> {
    let items = self.to_vec();
    items.iter().rposition(|x| x == item)
  }
}

impl<T: Clone + fmt::Display> FingerVec<T> {
  /// Debug view of the layout: `{front} tree {back}`.
  pub fn format_inline(&self) -> String {
    let mut out = String::new();
    out.push('{');
    for (i, x) in self.front.as_slice().iter().enumerate() {
      if i > 0 {
        out.push(' ');
      }
      out.push_str(&x.to_string());
    }
    out.push_str("} ");
    match &self.tree {
      Some(t) => t.format_inline(&mut out),
      None => out.push('_'),
    }
    out.push_str(" {");
    for (i, x) in self.back.as_slice().iter().enumerate() {
      if i > 0 {
        out.push(' ');
      }
      out.push_str(&x.to_string());
    }
    out.push('}');
    out
  }
}

// ---------- iterator ----------

/// Borrowing iterator over a [`FingerVec`]. Forward iteration walks the
/// chunks in order; `next_back` indexes from the end.
pub struct Iter<'a, T> {
  source: &'a FingerVec<T>,
  lo: usize,
  hi: usize,
  front: std::slice::Iter<'a, T>,
  tree: Option<&'a Tree<T>>,
  stack: Vec<(&'a [Tree<T>], usize)>,
  current: std::slice::Iter<'a, T>,
  back: std::slice::Iter<'a, T>,
}

impl<'a, T: Clone> Iter<'a, T> {
  fn new(v: &'a FingerVec<T>) -> Self {
    Iter {
      source: v,
      lo: 0,
      hi: v.len(),
      front: v.front.as_slice().iter(),
      tree: v.tree.as_ref(),
      stack: Vec::new(),
      current: [].iter(),
      back: v.back.as_slice().iter(),
    }
  }

  fn next_leaf(&mut self) -> Option<&'a [T]> {
    if let Some(t) = self.tree.take() {
      match t {
        Tree::Leaf(v) => return Some(v),
        Tree::Branch(b) => self.stack.push((&b.children, 0)),
      }
    }
    while let Some((children, i)) = self.stack.last_mut() {
      if *i >= children.len() {
        self.stack.pop();
        continue;
      }
      let child = &children[*i];
      *i += 1;
      match child {
        Tree::Leaf(v) => return Some(v),
        Tree::Branch(b) => self.stack.push((&b.children, 0)),
      }
    }
    None
  }
}

impl<'a, T: Clone> Iterator for Iter<'a, T> {
  type Item = &'a T;

  #[inline]
  fn next(&mut self) -> Option<&'a T> {
    if self.lo >= self.hi {
      return None;
    }
    self.lo += 1;
    if let Some(x) = self.front.next() {
      return Some(x);
    }
    loop {
      if let Some(x) = self.current.next() {
        return Some(x);
      }
      match self.next_leaf() {
        Some(leaf) => self.current = leaf.iter(),
        None => return self.back.next(),
      }
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let n = self.hi - self.lo;
    (n, Some(n))
  }
}

impl<'a, T: Clone> DoubleEndedIterator for Iter<'a, T> {
  fn next_back(&mut self) -> Option<&'a T> {
    if self.lo >= self.hi {
      return None;
    }
    self.hi -= 1;
    self.source.get(self.hi)
  }
}

impl<T: Clone> ExactSizeIterator for Iter<'_, T> {}

impl<'a, T: Clone> IntoIterator for &'a FingerVec<T> {
  type Item = &'a T;
  type IntoIter = Iter<'a, T>;
  fn into_iter(self) -> Iter<'a, T> {
    self.iter()
  }
}

// ---------- conversions ----------

impl<T: Clone> From<Vec<T>> for FingerVec<T> {
  fn from(xs: Vec<T>) -> Self {
    if xs.len() <= CHUNK {
      return FingerVec {
        front: Buf::empty(),
        tree: None,
        back: Buf::from_vec(xs),
      };
    }
    FingerVec {
      front: Buf::empty(),
      tree: build_from_chunks(chunk_values(xs)),
      back: Buf::empty(),
    }
  }
}

impl<T: Clone> From<&Vec<T>> for FingerVec<T> {
  fn from(xs: &Vec<T>) -> Self {
    Self::from(xs.to_owned())
  }
}

impl<T: Clone> From<&[T]> for FingerVec<T> {
  fn from(xs: &[T]) -> Self {
    Self::from(xs.to_vec())
  }
}

impl<T: Clone, const N: usize> From<&[T; N]> for FingerVec<T> {
  fn from(xs: &[T; N]) -> Self {
    Self::from(xs.to_vec())
  }
}

impl<T: Clone> FromIterator<T> for FingerVec<T> {
  fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
    Self::from(iter.into_iter().collect::<Vec<T>>())
  }
}

// ---------- comparisons ----------

impl<T: Clone + PartialEq> PartialEq for FingerVec<T> {
  fn eq(&self, other: &Self) -> bool {
    self.len() == other.len() && self.iter().eq(other.iter())
  }
}

impl<T: Clone + Eq> Eq for FingerVec<T> {}

impl<T: Clone + PartialOrd> PartialOrd for FingerVec<T> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.iter().partial_cmp(other.iter())
  }
}

/// Lexicographic order, the same as `Vec<T>`.
impl<T: Clone + Ord> Ord for FingerVec<T> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.iter().cmp(other.iter())
  }
}

/// Hashes the length and the elements, the same as `[T]`.
impl<T: Clone + Hash> Hash for FingerVec<T> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    state.write_usize(self.len());
    for x in self.iter() {
      x.hash(state);
    }
  }
}

impl<T: Clone> Index<usize> for FingerVec<T> {
  type Output = T;
  fn index(&self, idx: usize) -> &T {
    match self.get(idx) {
      Some(x) => x,
      None => panic!("index {idx} out of bounds for length {}", self.len()),
    }
  }
}

impl<T: Clone + fmt::Display> fmt::Display for FingerVec<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "(FingerVec")?;
    for x in self.iter() {
      write!(f, " {x}")?;
    }
    write!(f, ")")
  }
}
