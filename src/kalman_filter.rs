use nalgebra::{ArrayStorage, Const, Matrix, Vector};



pub type SimpleVector<const SIZE: usize> = Vector<f64, Const<SIZE>, ArrayStorage<f64, SIZE, 1>>;
pub type SimpleSquareMatrix<const SIZE: usize> = Matrix<f64, Const<SIZE>, Const<SIZE>, ArrayStorage<f64, SIZE, SIZE>>;



/// **Implements a kalman filter over an arbitrary number of variables.**
/// 
/// This can be used to track the state of the system. The filter is based on three factors: 
/// past values of the system, knowledge of how the system will evolve, and new measurements.
/// By using not just estimates of these values, but also precise descriptions of how accurate
/// each one is, the filter gives highly accurate data about the state of the system, including
/// how accurate that data is.
/// ## Usage
/// 
/// ### Inputs:
/// *On creation*
/// * Initial state
/// * Initial covariance matrix (reliability) of initial state
/// * Number of variables included (as generic)
/// 
/// *Throughout, binding must be given at start*
/// * Estimate of next state, given time change, and current state and variance
/// * Estimate of next variance, given time change, and current state and variance
/// 
/// *Throughout, as step inputs*
/// * New measurements
/// * Covariance matrices of measurements
/// * Changes in time from one step to next
/// 
/// ### Outputs:
/// * State estimate at current time
/// * Variance of state estimate at current time
/// 
/// ### Usage steps:
/// * Initialize and store
/// * Use .step_time() to step time forward (mutates state)
/// * Include new measurements using .apply_measurements (mutates state)
///   (this can be combined with the previous state using .step_with_measurement)
/// * Extract state with .get_current_state() and .get_current_covariance()
/// 
/// ## Notes
/// ### See also
/// TODO: linear version.
/// ### Limitations
/// **Distributions**
/// To make the math possible, the error in all measurements and predictions is assumed
/// to be gaussian. In the real world, this is often a good approximation, but rarely exactly right.
/// ### Further reading
/// * Kalman filters - https://en.wikipedia.org/wiki/Kalman_filter
/// * Covariance matrix - https://en.wikipedia.org/wiki/Covariance_matrix
/// * Multidimensional normal distribution - https://en.wikipedia.org/wiki/Multivariate_normal_distribution
pub struct KalmanFilter<const STATE_FACTORS: usize> {
    current_state: SimpleVector<STATE_FACTORS>,
    current_covariance: SimpleSquareMatrix<STATE_FACTORS>,
    evolution_function: Box<dyn Fn(
        SimpleVector<STATE_FACTORS>, 
        SimpleSquareMatrix<STATE_FACTORS>, 
        f64
    ) -> (
        SimpleVector<STATE_FACTORS>, 
        SimpleSquareMatrix<STATE_FACTORS>
    )>,
}



impl<const STATE_FACTORS: usize> KalmanFilter<STATE_FACTORS> {
    /// Creates a new KalmanFilter from the starting state and variance of
    /// the system, along with a function describing how future state should
    /// be reckoned.
    pub fn new(
        current_state: SimpleVector<STATE_FACTORS>,
        current_covariance: SimpleSquareMatrix<STATE_FACTORS>,
        evolution_function: impl Fn(
            SimpleVector<STATE_FACTORS>, 
            SimpleSquareMatrix<STATE_FACTORS>, 
            f64
        ) -> (
            SimpleVector<STATE_FACTORS>, 
            SimpleSquareMatrix<STATE_FACTORS>
        ) + 'static,
    ) -> Self {
        Self {
            current_state,
            current_covariance,
            evolution_function:Box::new(evolution_function)
        }
    }


    /// Pushes the state of the filter forward by some time. Mutates the object.
    /// Requires a time step, measurement, and the variance in that measurement.
    pub fn step_with_measurement(
        &mut self,
        dt: f64, 
        measurement: &SimpleVector<STATE_FACTORS>,
        measurement_covariance: &SimpleSquareMatrix<STATE_FACTORS>
    ) {
        self.step_time(dt);
        self.apply_measurement(measurement, measurement_covariance);
    }


    /// Move time forward by a specified value in the model. This involves using the
    /// evolution function provided at construction.
    /// NOTE: calling this function two times with dt=0.5 may not be the same as
    /// calling it once with dt=1. This is only dependent on the evolution function.
    pub fn step_time(&mut self, dt: f64) {
        (self.current_state, self.current_covariance) = (self.evolution_function)(
            self.current_state,
            self.current_covariance,
            dt
        );
    }


    /// Apply newly collected data to the model. This involves merging the current
    /// predicted state of the system with the new data. This function does not include
    /// the passage of time. If this is desired, use .step_time or .step_with_measurement.
    pub fn apply_measurement(
        &mut self,
        measurement: &SimpleVector<STATE_FACTORS>,
        measurement_covariance: &SimpleSquareMatrix<STATE_FACTORS>
    ) {
        (self.current_state, self.current_covariance) = Self::combine_measurements(
            &measurement,
            &measurement_covariance,
            &self.current_state,
            &self.current_covariance,
        );
    }


    /// Returns the current best estimate of the state of the system.
    pub fn get_current_state(&self) -> SimpleVector<STATE_FACTORS> {
        self.current_state
    }
    /// Returns the covariance matrix describing the uncertainty in the state of the system.
    pub fn get_current_covariance(&self) -> SimpleSquareMatrix<STATE_FACTORS> {
        self.current_covariance
    }


    /// Takes two measurements, with their variances, and determines the probability distribution
    /// of what the true value is likely to be.
    /// 
    /// The math for this is fairly crazy. You have been warned.
    /// The current implementation uses this math 
    /// (https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions),
    /// although it doesn't match numerical solutions. The prior and marginal probability
    /// are currently assumed to be improper flat distributions.
    fn combine_measurements(
        mean_1: &SimpleVector<STATE_FACTORS>,
        covariance_1: &SimpleSquareMatrix<STATE_FACTORS>,
        mean_2: &SimpleVector<STATE_FACTORS>,
        covariance_2: &SimpleSquareMatrix<STATE_FACTORS>,
    ) -> (SimpleVector<STATE_FACTORS>, SimpleSquareMatrix<STATE_FACTORS>) {

        let inverse_sum = 
            (covariance_1 + covariance_2).try_inverse().expect("Singular covariance matrices are not allowed.");

        (
            covariance_2 * inverse_sum * mean_1 + covariance_1 * inverse_sum * mean_2,
            covariance_1 * inverse_sum * covariance_2,
        )
        
    }
}