## FingerVec

A persistent (immutable, structurally shared) vector built for the [Calcit](https://github.com/calcit-lang/calcit) runtime. It replaces [ternary-tree](https://github.com/calcit-lang/ternary-tree.rs) as Calcit's list backend. The TypeScript twin lives in [finger-vec.ts](https://github.com/calcit-lang/finger-vec.ts).

[![Crates.io](https://img.shields.io/crates/v/finger_vec?style=flat-square)](https://crates.io/crates/finger_vec)

### Layout

```text
front buffer | relaxed B-tree of chunks | back buffer
```

- **Chunks**: elements are stored in arrays of up to 32 items, so a list of 32 items or fewer is a single buffer.
- **Relaxed B-tree**: every leaf sits at the same depth and branches keep cumulative sizes, as in RRB vectors. Branching factor is 32.
- **Buffers at both ends**: pushing and popping at either end touches a buffer; a full buffer moves into the tree as one chunk. This is the "finger" in the name, after finger trees, which keep the ends of a sequence cheap. It is not a finger tree.
- **Concatenation joins along the seam** and merges or rebalances the nodes there, so concatenating in a loop keeps the height logarithmic. Splits and concatenations that would leave the tree taller than a bound rebuild it.

| Operation | Cost |
| --- | --- |
| `get`, `assoc` | `O(log₃₂ n)` |
| `push_left`, `push_right`, `drop_left`, `drop_right` | amortized `O(1)` plus a chunk copy |
| `concat`, `split`, `slice`, `insert`, `dissoc` | `O(log n)` |
| `From<Vec<T>>` | `O(n)`, moves elements without cloning |

### Usage

```rust
use finger_vec::FingerVec;

let xs = FingerVec::from(vec![1, 2, 3]);
let ys = xs.push_right(4).push_left(0);
assert_eq!(ys.to_vec(), vec![0, 1, 2, 3, 4]);
assert_eq!(xs.len(), 3); // the original is unchanged

let (left, right) = ys.split(2);
assert_eq!(left.concat_with(&right), ys);
assert_eq!(ys.slice(1, 3).unwrap().to_vec(), vec![1, 2]);
```

`FingerVec` also provides the method names Calcit used with ternary-tree (`push`, `prepend`, `rest`, `butlast`, `skip`, `take`, `assoc_before`, `assoc_after`, `traverse`, `traverse_result`, `format_inline`), so switching is mostly a rename. Equality, ordering and hashing follow `Vec<T>`: lexicographic order, and the hash covers length and elements.

### Performance

Measured against `im_ternary_tree 0.0.21` with `usize` elements, release build, Linux x86-64, single runs:

| Case | ternary-tree | FingerVec | Speedup |
| --- | ---: | ---: | ---: |
| `push_right` 100k | 19.65 ms | 12.65 ms | 1.6× |
| `push_left` 100k | 33.07 ms | 22.42 ms | 1.5× |
| from `Vec` 100k | 19.92 ms | 2.22 ms | 9.0× |
| random `get` 100k | 37.06 ms | 8.19 ms | 4.5× |
| iterate 100k | 2.45 ms | 0.30 ms | 8.1× |
| `drop_left` until empty, 100k | 59.89 ms | 8.70 ms | 6.9× |
| random `assoc` 100k | 253.67 ms | 119.93 ms | 2.1× |
| `concat acc [x]` 5k, then `get` every item | 49.97 ms | 0.10 ms | ~500× |
| insert in the middle 5k | 592.14 ms | 19.06 ms | 31× |
| queue: `push_right` + `drop_left` 100k | 39.78 ms | 17.05 ms | 2.3× |
| 32-item list: `push_right` ×1M | 351.11 ms | 86.79 ms | 4.0× |
| 32-item list: from `Vec` ×1M | 1553.52 ms | 37.76 ms | 41× |
| `slice(1, len)` repeated 20k | 17.01 ms | 20.99 ms | 0.8× |

The ternary tree degrades into a chain under repeated `concat` or middle insertion (depth 4,999 after 5,000 concatenations); FingerVec stays at depth 3–4. Run `cargo run --release --example bench` for FingerVec numbers on your machine.

### Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --example bench
```

Tests compare random operation sequences against `Vec<T>`, check structural invariants after every step, and assert depth bounds for concatenation loops, middle insertion and queue usage.

### License

MIT
