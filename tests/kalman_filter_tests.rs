use kalman_filter::*;
use nalgebra::*;

fn assert_float_eq(a:f64, b:f64, e: f64) {
	assert!((a - b).abs() < e);
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
		Vector1::new(2.0), 
		Matrix1::new(2.0)
	);

	assert_float_eq(*filter.get_current_state().index(0), 0.4, 1e-6);
	assert_float_eq(*filter.get_current_covariance().index(0), 0.894427191, 1e-6);

	// Step 2
	filter.step(
		0.1, 
		Vector1::new(2.0), 
		Matrix1::new(2.0)
	);

	assert_float_eq(*filter.get_current_state().index(0), 0.666666666667, 1e-6);
	assert_float_eq(*filter.get_current_covariance().index(0), 0.816496580928, 1e-6);
}