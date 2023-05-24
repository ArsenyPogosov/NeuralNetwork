#include <activation_functions.h>

#include <cmath>

namespace neural::activation
{

Eigen::VectorXd SoftMax::Eval0(const Eigen::VectorXd &x) const
{
	Eigen::VectorXd res = x.array().exp();
	double l = res.sum();
	return res / l;
}

Eigen::MatrixXd SoftMax::Eval1(const Eigen::VectorXd &x) const
{
	Eigen::MatrixXd s = Eval0(x).replicate(1, x.rows());

	return s.cwiseProduct(Eigen::MatrixXd::Identity(x.rows(), x.rows()) -
	                      s.transpose());
}

}  // namespace neural::activation
