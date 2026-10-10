use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn masked_argmax(logits: PyReadonlyArray1<'_, f32>, mask: PyReadonlyArray1<'_, i32>) -> usize {
    let logits: ArrayView1<'_, f32> = logits.as_array();
    let mask: ArrayView1<'_, i32> = mask.as_array();
    let mut best_token = 0;
    let mut best_value = f32::NEG_INFINITY;

    for (token, &value) in logits.iter().enumerate() {
        let word = token / 32;
        let bit = token % 32;
        let allowed = word < mask.len() && ((mask[word] as u32) & (1_u32 << bit)) != 0;
        let is_better = value > best_value || (value.is_nan() && !best_value.is_nan());
        if allowed && is_better {
            best_token = token;
            best_value = value;
        }
    }
    best_token
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(masked_argmax, m)?)?;
    Ok(())
}
