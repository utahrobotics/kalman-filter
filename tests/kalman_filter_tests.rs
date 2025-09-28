use kalman_filter::*;
use nalgebra::*;

fn assert_float_eq(expected:f64, actual:f64, margin: f64) {
	assert!(
		(expected - actual).abs() < margin, 
		"assertion failed: Expected {expected}, but found {actual} (margin of {margin})"
	);
}

/// Tests the kalman filter's basic distribution combining
/// ability. Does not test proper affect of evolution function.
/// Uses one dimension. Check values determined externally.
/// See https://www.desmos.com/calculator/hp8y5nueim for calculations.
#[test]
fn simple_1d() {
	// Setup
	//   Assume value never changes for prediction func
	fn evolution_func(
        state: SimpleVector<1>, 
        covar: SimpleSquareMatrix<1>, 
        _: f64
    ) -> (
        SimpleVector<1>, 
        SimpleSquareMatrix<1>
    ) {
		(state, covar)
	}

	let mut filter = KalmanFilter::new(
		Vector1::new(0.0), 
		Matrix1::new(1.0),
		evolution_func
	);

	// Check initial filter state
	assert_eq!(*filter.get_current_state().index(0), 0.0);
	assert_eq!(*filter.get_current_covariance().index(0), 1.0);

	// Step 1
	filter.step(
		0.1, 
		&Vector1::new(2.0), 
		&Matrix1::new(2.0)
	);

	assert_float_eq(0.6666666666666666, *filter.get_current_state().index(0), 1e-6);
	assert_float_eq(0.6666666666666666, *filter.get_current_covariance().index(0), 1e-6);

	// Step 2
	filter.step(
		0.1, 
		&Vector1::new(2.0), 
		&Matrix1::new(2.0)
	);

	assert_float_eq(1.0, *filter.get_current_state().index(0), 1e-6);
	assert_float_eq(0.5, *filter.get_current_covariance().index(0), 1e-6);
}