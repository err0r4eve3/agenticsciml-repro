# Piecewise Gated Experts

For a target with different behavior on each side of an interface, train
separate local experts and combine them with a bounded gate. Keep the gate
stable so optimization does not collapse into a single expert.
