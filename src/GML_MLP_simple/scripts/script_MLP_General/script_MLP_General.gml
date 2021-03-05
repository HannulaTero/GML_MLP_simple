

/// @func	Tanh(input);
/// @desc	Squishes value to be between -1 and +1 as S-curve. Similiar to Sigmoid
function Tanh(input) {
	return ((2 / (1 + exp(-2 * input))) - 1);
}

/// @func	TanhDerivative(input);
function TanhDerivative(input) {
	return (1 - sqr(Tanh(input)));
}
