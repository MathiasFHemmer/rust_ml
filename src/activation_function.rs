use std::f64::consts::E;

pub struct ActivationFuncion<'a> {
	pub function: &'a dyn Fn(f64) -> f64,
	pub derivative: &'a dyn Fn(f64) -> f64,
}

impl<'a> Clone for ActivationFuncion<'a> {
    fn clone(&self) -> Self {
        Self { function: self.function.clone(), derivative: self.derivative.clone() }
    }
}

pub const IDENTITY: ActivationFuncion = ActivationFuncion {
	function: &|x| x,
	derivative: &|_| 1.0,
};

pub const SIGMOID: ActivationFuncion = ActivationFuncion {
	function: &|x| 1.0 / (1.0 + E.powf(-x)),
	derivative: &|x| x * (1.0 - x),
};
