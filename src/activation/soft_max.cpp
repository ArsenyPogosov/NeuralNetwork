#include <activation_functions.h>

#include <cmath>

namespace neural::activation
{

Eigen::VectorXd SoftMax::Eval0(Eigen::VectorXd x) const
{
	x = x.array().exp();
	double l = x.sum();
	return x / l;
}

Eigen::MatrixXd SoftMax::Eval1(Eigen::VectorXd x) const
{
	Eigen::MatrixXd s = Eval0(x).replicate(1, x.rows());

	return s.cwiseProduct(Eigen::MatrixXd::Identity(x.rows(), x.rows()) -
	                      s.transpose());
}

}  // namespace neural::activation
