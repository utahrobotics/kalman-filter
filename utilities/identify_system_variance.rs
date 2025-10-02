use csv::ReaderBuilder;
use kalman_filter::SimpleSquareMatrix;
use std::{env, error::Error};



/// Turns a set of measurements into a mean and covariance matrix.
/// The current implementation is locked to a 3 vector. If you want to
/// figure out a good way of extending it, be my guest.
fn main() -> Result<(), Box<dyn Error>> {
	let path = env::args().nth(1).expect("File path must be passed as an argument.");
	let mut reader = ReaderBuilder::new()
		.has_headers(false)
		.from_path(path)?
	;

	let rows: Vec<([f64;3], [f64;3])> = reader
		.deserialize::<([f64; 3], [f64; 3])>()
		.map(|x| x.expect("Deserialization failed"))
		.collect()
	;
	
	let mut covariance_matrix: SimpleSquareMatrix<3> = SimpleSquareMatrix::zeros();
	for (actual, measurement) in &rows {
		for i in 0..3 {
			for j in 0..3 {
				*covariance_matrix.index_mut(i + 3*j) += (measurement[i] - actual[i]) * (measurement[j] - actual[j]);
			}
		}
	}

	covariance_matrix.scale_mut(1.0 / rows.len() as f64);

	println!("{}", covariance_matrix);

	Ok(())
}