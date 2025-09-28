use crate::kalman_filter::*;

// TODO: (from issue #2)
// * Replacing evolution function pointer with matrix.
// * Replacing full covariance matrix with diagonal covariance matrix.
// * Allowing a constant dt to be set on creation of filter struct.



/// Create a covariance matrix from the variances of each variable, stored in
/// a vector. This assumes the variables are completely independent.
pub fn covariance_matrix_from_variance_vector<const SIZE: usize>(variances: &SimpleVector<SIZE>) -> SimpleSquareMatrix<SIZE> {
	SimpleSquareMatrix::from_diagonal(variances)
}

/// Create a covariance matrix from one variance value. This assumes the 
/// variables are completely independent, and equally variant.
pub fn covariance_matrix_from_variance<const SIZE: usize>(variance: f64) -> SimpleSquareMatrix<SIZE> {
	SimpleSquareMatrix::from_diagonal_element(variance)
}
