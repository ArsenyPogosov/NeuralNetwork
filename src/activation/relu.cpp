#include <activation_functions.h>

namespace neural::activation
{

Eigen::VectorXd ReLu::Eval0(Eigen::VectorXd x) const
{
	return x.array().max(Eigen::ArrayXd::Zero(x.rows(), 1));
}

Eigen::MatrixXd ReLu::Eval1(Eigen::VectorXd x) const
{
	return x.unaryExpr([](double x) -> double { return x > 0; }).asDiagonal();
}

}  // namespace neural::activation
