use csv::ReaderBuilder;
use kalman_filter::SimpleSquareMatrix;
use std::{env, error::Error};

const VEC_SIZE: usize = 3;



/// Turns a set of measurements into a mean and covariance matrix.
/// The current implementation is locked to a hardcoded vector. If you want to
/// figure out a good way of extending it, be my guest.
fn main() -> Result<(), Box<dyn Error>> {
	let path = env::args().nth(1).expect("File path must be passed as an argument.");
	let mut reader = ReaderBuilder::new()
		.has_headers(false)
		.from_path(path)?
	;

	let rows: Vec<([f64;VEC_SIZE], [f64;VEC_SIZE])> = reader
		.deserialize::<([f64; VEC_SIZE], [f64; VEC_SIZE])>()
		.map(|x| x.expect("Deserialization failed"))
		.collect()
	;
	
	let mut covariance_matrix: SimpleSquareMatrix<VEC_SIZE> = SimpleSquareMatrix::zeros();
	for (actual, measurement) in &rows {
		for i in 0..VEC_SIZE {
			for j in 0..VEC_SIZE {
				*covariance_matrix.index_mut(i + VEC_SIZE*j) += (measurement[i] - actual[i]) * (measurement[j] - actual[j]);
			}
		}
	}

	covariance_matrix.scale_mut(1.0 / rows.len() as f64);

	println!("{}", covariance_matrix);

	Ok(())
}