//! Model-independent native kernels. C callers own input buffers for each call.
//! An Index is immutable after construction: searches may run concurrently.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const ABI_VERSION: u32 = 1;
pub const MAX_STEPS: usize = 32;

#[derive(Debug)]
pub struct Index {
    data: Vec<f32>,
    rows: usize,
    dim: usize,
}

#[derive(Clone, Copy, Debug)]
struct Hit {
    row: usize,
    score: f64,
}
impl PartialEq for Hit {
    fn eq(&self, b: &Self) -> bool {
        self.row == b.row && self.score == b.score
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
// The worst retained hit is at the heap root; lower row wins ties.
impl Ord for Hit {
    fn cmp(&self, b: &Self) -> Ordering {
        b.score.total_cmp(&self.score).then(self.row.cmp(&b.row))
    }
}

impl Index {
    pub fn new(data: &[f32], rows: usize, dim: usize) -> Result<Self, String> {
        if dim == 0
            || dim > 65536
            || rows.checked_mul(dim) != Some(data.len())
            || data.len() > 268435456
        {
            return Err("invalid vector shape".into());
        }
        let mut normalized = data.to_vec();
        for row in normalized.chunks_mut(dim) {
            let norm = row
                .iter()
                .try_fold(0.0_f64, |s, &x| {
                    if x.is_finite() {
                        Some(s + (x as f64) * (x as f64))
                    } else {
                        None
                    }
                })
                .ok_or("non-finite vector")?
                .sqrt();
            if norm == 0.0 {
                return Err("zero vector".into());
            }
            for x in row {
                *x = (*x as f64 / norm) as f32;
            }
        }
        Ok(Self {
            data: normalized,
            rows,
            dim,
        })
    }
    pub fn search(
        &self,
        query: &[f32],
        mask: &[u8],
        k: usize,
    ) -> Result<Vec<(usize, f64)>, String> {
        if query.len() != self.dim || mask.len() != self.rows || k == 0 || k > 10000 {
            return Err("invalid query shape or k".into());
        }
        let norm = query
            .iter()
            .try_fold(0.0_f64, |s, &x| {
                if x.is_finite() {
                    Some(s + (x as f64) * (x as f64))
                } else {
                    None
                }
            })
            .ok_or("non-finite query")?
            .sqrt();
        if norm == 0.0 {
            return Err("zero query".into());
        }
        let q: Vec<f64> = query.iter().map(|&x| x as f64 / norm).collect();
        let mut heap: BinaryHeap<Hit> = BinaryHeap::with_capacity(k + 1);
        for (i, row) in self.data.chunks_exact(self.dim).enumerate() {
            if mask[i] == 0 {
                continue;
            }
            let score = row.iter().zip(&q).map(|(&x, &y)| x as f64 * y).sum();
            let h = Hit { row: i, score };
            if heap.len() < k {
                heap.push(h);
            } else if h < *heap.peek().unwrap() {
                heap.pop();
                heap.push(h);
            }
        }
        let mut result: Vec<_> = heap.into_iter().map(|h| (h.row, h.score)).collect();
        result.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(result)
    }
}

fn atom(x: &str) -> bool {
    !x.is_empty()
        && x.split(['.', '-']).all(|p| {
            !p.is_empty()
                && p.bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        })
}
fn number(x: &str) -> bool {
    !x.is_empty() && (x == "0" || !x.starts_with('0')) && x.bytes().all(|c| c.is_ascii_digit())
}
pub fn valid_uri(uri: &str) -> bool {
    let Some(rest) = uri.strip_prefix("proc://") else {
        return false;
    };
    let p: Vec<_> = rest.split('/').collect();
    if uri.len() > 512 || p.len() != 4 || !p[..3].iter().all(|p| atom(p)) {
        return false;
    }
    if let Some(v) = p[3].strip_prefix('v') {
        number(v)
    } else {
        let v: Vec<_> = p[3].split('.').collect();
        v.len() == 3 && v.iter().all(|p| number(p))
    }
}
pub fn valid_step(id: &str) -> bool {
    let mut b = id.bytes();
    id.len() <= 64
        && b.next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && b.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}
fn hash(x: &str) -> bool {
    x.len() == 64
        && x.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub format: String,
    pub catalog_revision: String,
    pub steps: Vec<Step>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub id: String,
    pub uri: String,
    pub digest: String,
    pub args: serde_json::Map<String, Value>,
}

pub fn validate_plan(plan: &Plan) -> Result<(), String> {
    if plan.format != "nlbridge/plan-v1"
        || !hash(&plan.catalog_revision)
        || plan.steps.is_empty()
        || plan.steps.len() > MAX_STEPS
    {
        return Err("invalid plan header".into());
    }
    let mut seen = HashSet::new();
    for step in &plan.steps {
        if !valid_step(&step.id)
            || seen.contains(step.id.as_str())
            || !valid_uri(&step.uri)
            || !hash(&step.digest)
        {
            return Err("invalid or duplicate step".into());
        }
        for value in step.args.values() {
            if let Some(obj) = value.as_object() {
                if let Some(reference) = obj.get("$ref") {
                    if obj.len() != 1 {
                        return Err("reference has extra keys".into());
                    }
                    let r = reference.as_object().ok_or("reference must be object")?;
                    let target = r
                        .get("step")
                        .and_then(Value::as_str)
                        .ok_or("missing reference step")?;
                    let pointer = r
                        .get("pointer")
                        .and_then(Value::as_str)
                        .ok_or("missing reference pointer")?;
                    if r.len() != 2
                        || !seen.contains(target)
                        || !(pointer.is_empty() || pointer.starts_with('/'))
                    {
                        return Err("invalid, cyclic, or forward reference".into());
                    }
                    let bytes = pointer.as_bytes();
                    for i in 0..bytes.len() {
                        if bytes[i] == b'~'
                            && (i + 1 == bytes.len() || !matches!(bytes[i + 1], b'0' | b'1'))
                        {
                            return Err("invalid JSON pointer escape".into());
                        }
                    }
                }
            }
        }
        seen.insert(step.id.as_str());
    }
    Ok(())
}
pub fn render_plan(plan: &Plan) -> Result<String, String> {
    validate_plan(plan)?;
    let mut out = String::from("# nlbridge/dsl-v1\n");
    for step in &plan.steps {
        out.push_str(&format!(
            "{} = call({}, {});\n",
            step.id,
            serde_json::to_string(&step.uri).unwrap(),
            serde_json::to_string(&step.args).unwrap()
        ));
    }
    Ok(out)
}

#[no_mangle]
pub extern "C" fn nlb_abi_version() -> u32 {
    ABI_VERSION
}

/// # Safety
/// `data` must point to rows*dim readable f32 values for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn nlb_index_new(data: *const f32, rows: usize, dim: usize) -> *mut Index {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(len) = rows.checked_mul(dim) else {
            return std::ptr::null_mut();
        };
        if data.is_null() || len > 268435456 {
            return std::ptr::null_mut();
        }
        match Index::new(std::slice::from_raw_parts(data, len), rows, dim) {
            Ok(x) => Box::into_raw(Box::new(x)),
            Err(_) => std::ptr::null_mut(),
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}
/// # Safety
/// Pointer must be an unfreed pointer returned by nlb_index_new. No concurrent searches during free.
#[no_mangle]
pub unsafe extern "C" fn nlb_index_free(index: *mut Index) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

/// # Safety
/// Valid immutable index, query_len f32s, mask_len u8s, and two nonoverlapping writable output arrays of k elements.
#[no_mangle]
pub unsafe extern "C" fn nlb_search(
    index: *const Index,
    query: *const f32,
    query_len: usize,
    mask: *const u8,
    mask_len: usize,
    k: usize,
    out_rows: *mut usize,
    out_scores: *mut f64,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if index.is_null()
            || query.is_null()
            || mask.is_null()
            || out_rows.is_null()
            || out_scores.is_null()
        {
            return -1;
        }
        let index = &*index;
        if query_len != index.dim || mask_len != index.rows || k == 0 || k > 10000 {
            return -1;
        }
        match index.search(
            std::slice::from_raw_parts(query, query_len),
            std::slice::from_raw_parts(mask, mask_len),
            k,
        ) {
            Ok(hits) => {
                for (j, (row, score)) in hits.iter().enumerate() {
                    *out_rows.add(j) = *row;
                    *out_scores.add(j) = *score;
                }
                hits.len() as isize
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-2)
}

/// # Safety
/// `input` must be a readable NUL-terminated UTF-8 JSON string. Free result using nlb_string_free.
#[no_mangle]
pub unsafe extern "C" fn nlb_render_json(input: *const c_char) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() {
            return std::ptr::null_mut();
        }
        let bytes = CStr::from_ptr(input).to_bytes();
        if bytes.len() > 1048576 {
            return std::ptr::null_mut();
        }
        let result = serde_json::from_slice::<Plan>(bytes)
            .map_err(|e| e.to_string())
            .and_then(|p| render_plan(&p));
        let value = match result {
            Ok(dsl) => serde_json::json!({"ok":true,"dsl":dsl}),
            Err(e) => serde_json::json!({"ok":false,"error":e}),
        };
        CString::new(value.to_string()).unwrap().into_raw()
    }))
    .unwrap_or(std::ptr::null_mut())
}
/// # Safety
/// `value` must be an unfreed result of nlb_render_json, or null.
#[no_mangle]
pub unsafe extern "C" fn nlb_string_free(value: *mut c_char) {
    if !value.is_null() {
        drop(CString::from_raw(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn directions_and_versions() {
        assert!(valid_uri("proc://media/images/svg-to-png/v1"));
        for u in [
            "proc://media/images/svg-to-png/v01",
            "proc://a/b/c/01.2.3",
            "proc://a/b/c/v1;rm",
            "proc://a/b/../v1",
        ] {
            assert!(!valid_uri(u));
        }
    }
    #[test]
    fn mask_before_top_k() {
        let i = Index::new(&[1., 0., 1., 0., 0., 1.], 3, 2).unwrap();
        assert_eq!(i.search(&[1., 0.], &[0, 0, 1], 1).unwrap()[0].0, 2);
    }
    #[test]
    fn stable_ties() {
        let i = Index::new(&[1., 0., 1., 0.], 2, 2).unwrap();
        assert_eq!(i.search(&[1., 0.], &[1, 1], 1).unwrap()[0].0, 0);
    }
    #[test]
    fn reject_nan_zero() {
        assert!(Index::new(&[f32::NAN], 1, 1).is_err());
        assert!(Index::new(&[0.], 1, 1).is_err());
    }
    #[test]
    fn reject_invalid_shape() {
        assert!(Index::new(&[1.], 1, 2).is_err());
    }
    #[test]
    fn empty_index() {
        assert!(Index::new(&[], 0, 2)
            .unwrap()
            .search(&[1., 0.], &[], 1)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn graph_order() {
        let h = "a".repeat(64);
        let mut p: Plan=serde_json::from_value(serde_json::json!({"format":"nlbridge/plan-v1","catalog_revision":h,"steps":[{"id":"s1","uri":"proc://a/b/c/v1","digest":h,"args":{"x":{"$ref":{"step":"s2","pointer":"/x"}}}}]})).unwrap();
        assert!(validate_plan(&p).is_err());
        p.steps[0].args.clear();
        assert!(render_plan(&p).unwrap().contains("call("));
    }
}
