#include <activation_functions.h>

#include <cmath>

namespace neural::activation
{

Eigen::VectorXd Sigmoid::Eval0(Eigen::VectorXd x) const
{
	return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
}

Eigen::MatrixXd Sigmoid::Eval1(Eigen::VectorXd x) const
{
	Eigen::ArrayXd s = Eval0(x);
	s *= (1 - s);
	return s.matrix().asDiagonal();
}

}  // namespace neural::activation
