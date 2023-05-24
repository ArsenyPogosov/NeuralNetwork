#include <activation_functions.h>

namespace neural::activation
{

Eigen::VectorXd Linear::Eval0(const Eigen::VectorXd &x) const
{
	return x;
}

Eigen::MatrixXd Linear::Eval1(const Eigen::VectorXd &x) const
{
	return Eigen::MatrixXd::Identity(x.rows(), x.rows());
}

}  // namespace neural::activation
