//! Persistent JSON Lines interface: load vectors once, then issue searches.
use nlbridge_core::{render_plan, Index, Plan};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
fn run(v: Value, index: &mut Option<Index>) -> Result<Value, String> {
    match v.get("command").and_then(Value::as_str) {
        Some("load") => {
            let rows: Vec<Vec<f32>> =
                serde_json::from_value(v["vectors"].clone()).map_err(|e| e.to_string())?;
            let dim = rows.first().ok_or("empty matrix")?.len();
            if rows.iter().any(|r| r.len() != dim) {
                return Err("ragged matrix".into());
            }
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            *index = Some(Index::new(&flat, rows.len(), dim)?);
            Ok(json!({"ok":true,"rows":rows.len(),"dim":dim}))
        }
        Some("search") => {
            let q: Vec<f32> =
                serde_json::from_value(v["query"].clone()).map_err(|e| e.to_string())?;
            let mask: Vec<u8> =
                serde_json::from_value(v["mask"].clone()).map_err(|e| e.to_string())?;
            let k = v["k"].as_u64().ok_or("k required")? as usize;
            let hits = index
                .as_ref()
                .ok_or("load an index first")?
                .search(&q, &mask, k)?;
            Ok(json!({"ok":true,"hits":hits}))
        }
        Some("render") => {
            let p: Plan = serde_json::from_value(v["plan"].clone()).map_err(|e| e.to_string())?;
            Ok(json!({"ok":true,"dsl":render_plan(&p)?}))
        }
        _ => Err("commands: load, search, render".into()),
    }
}
fn main() {
    let mut index = None;
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let result = line
            .map_err(|e| e.to_string())
            .and_then(|s| {
                if s.len() > 1048576 {
                    Err("line too large".into())
                } else {
                    serde_json::from_str(&s).map_err(|e| e.to_string())
                }
            })
            .and_then(|v| run(v, &mut index));
        let value = match result {
            Ok(v) => v,
            Err(e) => json!({"ok":false,"error":e}),
        };
        if writeln!(out, "{value}").is_err() || out.flush().is_err() {
            break;
        }
    }
}
