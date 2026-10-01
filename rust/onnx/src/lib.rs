//! Optional ONNX embeddings. C callers must not free a handle in use.
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynValue, Tensor, TensorElementType, ValueType},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    path::PathBuf,
    ptr,
    sync::Mutex,
};
use tokenizers::Tokenizer;
pub const ABI_VERSION: u32 = 1;
static RUNTIME_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
type Result<T> = std::result::Result<T, String>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    runtime_library: String,
    model: String,
    tokenizer: String,
    output_name: String,
    pooling: String,
    dim: usize,
    max_length: usize,
    pad_id: u32,
    threads: usize,
}
struct Input {
    name: String,
    int32: bool,
}
pub struct Handle {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: Config,
    inputs: Vec<Input>,
}
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
impl Handle {
    fn open(config: Config) -> Result<Self> {
        if !(1..=65536).contains(&config.dim)
            || !(2..=8192).contains(&config.max_length)
            || !(1..=64).contains(&config.threads)
            || config.pad_id >= i32::MAX as u32
            || !["mean", "cls", "none"].contains(&config.pooling.as_str())
        {
            return Err("invalid ONNX configuration".into());
        }
        let path = std::fs::canonicalize(&config.runtime_library).map_err(err)?;
        {
            let mut active = RUNTIME_PATH.lock().map_err(err)?;
            if let Some(previous) = active.as_ref() {
                if previous != &path {
                    return Err("ONNX runtime library cannot change in a running process".into());
                }
            } else {
                ort::init_from(&path).map_err(err)?.commit();
                *active = Some(path);
            }
        }
        let session = Session::builder()
            .map_err(err)?
            .with_intra_threads(config.threads)
            .map_err(err)?
            .with_inter_threads(1)
            .map_err(err)?
            .with_parallel_execution(false)
            .map_err(err)?
            .with_intra_op_spinning(false)
            .map_err(err)?
            .with_inter_op_spinning(false)
            .map_err(err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(err)?
            .commit_from_file(&config.model)
            .map_err(err)?;
        let mut inputs = Vec::new();
        for input in session.inputs() {
            if !["input_ids", "attention_mask", "token_type_ids"].contains(&input.name()) {
                return Err(format!("unsupported ONNX input: {}", input.name()));
            }
            match input.dtype() {
                ValueType::Tensor { ty, shape, .. }
                    if shape.len() == 2
                        && (*ty == TensorElementType::Int64 || *ty == TensorElementType::Int32) =>
                {
                    inputs.push(Input {
                        name: input.name().into(),
                        int32: *ty == TensorElementType::Int32,
                    });
                }
                _ => return Err("ONNX inputs must be rank-2 int32/int64 tensors".into()),
            }
        }
        if !inputs.iter().any(|i| i.name == "input_ids")
            || !inputs.iter().any(|i| i.name == "attention_mask")
        {
            return Err("ONNX graph must accept input_ids and attention_mask".into());
        }
        let output = session
            .outputs()
            .iter()
            .find(|o| o.name() == config.output_name)
            .ok_or("ONNX output name not found")?;
        match output.dtype() {
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape,
                ..
            } if shape.len() == (if config.pooling == "none" { 2 } else { 3 }) => (),
            _ => return Err("ONNX output must be float32 with the configured pooling rank".into()),
        }
        let mut tokenizer = Tokenizer::from_file(&config.tokenizer).map_err(err)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(err)?;
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
            inputs,
        })
    }
    fn encode(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty()
            || texts.len() > 32
            || texts.iter().map(|s| s.len()).sum::<usize>() > 1 << 20
        {
            return Err("embedding batch must contain 1..32 texts within 1 MiB".into());
        }
        // Avoid a global Rayon pool competing with ORT inference workers.
        let encoded = texts
            .iter()
            .map(|t| self.tokenizer.encode(t.as_str(), true).map_err(err))
            .collect::<Result<Vec<_>>>()?;
        if encoded
            .iter()
            .any(|e| e.is_empty() || e.len() > self.config.max_length)
        {
            return Err(
                "input exceeds embedding token budget or tokenizes to empty; no text was truncated"
                    .into(),
            );
        }
        let batch = encoded.len();
        let seq = encoded.iter().map(|e| e.len()).max().unwrap();
        let mut ids = vec![i64::from(self.config.pad_id); batch * seq];
        let mut mask = vec![0_i64; batch * seq];
        let mut types = vec![0_i64; batch * seq];
        for (b, e) in encoded.iter().enumerate() {
            for t in 0..e.len() {
                ids[b * seq + t] = i64::from(e.get_ids()[t]);
                mask[b * seq + t] = i64::from(e.get_attention_mask()[t]);
                types[b * seq + t] = i64::from(e.get_type_ids()[t]);
            }
        }
        let mut values: Vec<(String, DynValue)> = Vec::new();
        for input in &self.inputs {
            let data = match input.name.as_str() {
                "input_ids" => &ids,
                "attention_mask" => &mask,
                _ => &types,
            };
            let value = if input.int32 {
                let converted = data
                    .iter()
                    .map(|v| i32::try_from(*v).map_err(err))
                    .collect::<Result<Vec<_>>>()?;
                Tensor::from_array(([batch, seq], converted))
                    .map_err(err)?
                    .into_dyn()
            } else {
                Tensor::from_array(([batch, seq], data.clone()))
                    .map_err(err)?
                    .into_dyn()
            };
            values.push((input.name.clone(), value));
        }
        let mut session = self.session.lock().map_err(err)?;
        let output = session.run(values).map_err(err)?;
        let (shape, data) = output[self.config.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(err)?;
        let expected = if self.config.pooling == "none" {
            vec![batch as i64, self.config.dim as i64]
        } else {
            vec![batch as i64, seq as i64, self.config.dim as i64]
        };
        if shape.as_ref() != expected.as_slice() {
            return Err("ONNX output dimensions mismatch".into());
        }
        pool(
            data,
            &mask,
            batch,
            seq,
            self.config.dim,
            &self.config.pooling,
        )
    }
}
fn pool(
    data: &[f32],
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
    mode: &str,
) -> Result<Vec<Vec<f32>>> {
    let mut result = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = vec![0_f64; dim];
        if mode == "none" || mode == "cls" {
            let offset = if mode == "none" {
                b * dim
            } else {
                b * seq * dim
            };
            for d in 0..dim {
                row[d] = f64::from(data[offset + d]);
            }
        } else {
            let count = mask[b * seq..(b + 1) * seq]
                .iter()
                .filter(|v| **v != 0)
                .count();
            if count == 0 {
                return Err("empty attention mask".into());
            }
            for t in 0..seq {
                if mask[b * seq + t] != 0 {
                    for d in 0..dim {
                        row[d] += f64::from(data[(b * seq + t) * dim + d]);
                    }
                }
            }
            for v in &mut row {
                *v /= count as f64;
            }
        }
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err("embedding is non-finite or zero".into());
        }
        result.push(row.into_iter().map(|v| (v / norm) as f32).collect());
    }
    Ok(result)
}
fn owned(value: Value) -> *mut c_char {
    CString::new(value.to_string()).unwrap().into_raw()
}
unsafe fn json_input<T: for<'de> Deserialize<'de>>(raw: *const c_char) -> Result<T> {
    if raw.is_null() {
        return Err("null input".into());
    }
    let bytes = CStr::from_ptr(raw).to_bytes();
    if bytes.len() > 2 << 20 {
        return Err("FFI JSON exceeds 2 MiB".into());
    }
    serde_json::from_slice(bytes).map_err(err)
}
#[no_mangle]
pub extern "C" fn nlb_onnx_abi_version() -> u32 {
    ABI_VERSION
}
/// # Safety
/// Config must be NUL-terminated; error must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn nlb_onnx_open(
    config: *const c_char,
    error: *mut *mut c_char,
) -> *mut Handle {
    if error.is_null() {
        return ptr::null_mut();
    }
    *error = ptr::null_mut();
    match catch_unwind(AssertUnwindSafe(|| Handle::open(json_input(config)?))) {
        Ok(Ok(h)) => Box::into_raw(Box::new(h)),
        failure => {
            let message = match failure {
                Ok(Err(e)) => e,
                _ => "panic while opening ONNX runtime".into(),
            };
            *error = owned(json!({"error":message}));
            ptr::null_mut()
        }
    }
}
/// # Safety
/// Handle must be live; texts must be NUL-terminated JSON. Free response with string_free.
#[no_mangle]
pub unsafe extern "C" fn nlb_onnx_embed(handle: *mut Handle, texts: *const c_char) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = handle.as_ref().ok_or("null ONNX handle")?;
        handle.encode(json_input(texts)?)
    }));
    owned(match result {
        Ok(Ok(vectors)) => json!({"vectors":vectors}),
        Ok(Err(e)) => json!({"error":e}),
        Err(_) => json!({"error":"panic during ONNX inference"}),
    })
}
/// # Safety
/// Handle must be live. Free response with string_free.
#[no_mangle]
pub unsafe extern "C" fn nlb_onnx_info(handle: *mut Handle) -> *mut c_char {
    owned(if let Some(h) = handle.as_ref() {
        json!({"runtime":ort::info(),"threads":h.config.threads,"dim":h.config.dim,"inputs":h.inputs.iter().map(|i| json!({"name":i.name,"dtype":if i.int32 {"int32"} else {"int64"}})).collect::<Vec<_>>()})
    } else {
        json!({"error":"null ONNX handle"})
    })
}
/// # Safety
/// Free once, after all calls with this handle complete.
#[no_mangle]
pub unsafe extern "C" fn nlb_onnx_free(handle: *mut Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
/// # Safety
/// Free only pointers returned by this library, exactly once.
#[no_mangle]
pub unsafe extern "C" fn nlb_onnx_string_free(text: *mut c_char) {
    if !text.is_null() {
        drop(CString::from_raw(text));
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn masked_pooling() {
        assert_eq!(
            pool(&[3., 4., 999., 999.], &[1, 0], 1, 2, 2, "mean").unwrap(),
            vec![vec![0.6, 0.8]]
        );
    }
    #[test]
    fn zero_and_nan_rejected() {
        assert!(pool(&[0., 0.], &[1], 1, 1, 2, "mean").is_err());
        assert!(pool(&[f32::NAN, 1.], &[1], 1, 1, 2, "cls").is_err());
        assert!(pool(&[1., 1.], &[0], 1, 1, 2, "mean").is_err());
    }
}
