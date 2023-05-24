#include <loss_functions.h>

#include <Eigen/Dense>

namespace neural::loss
{

double CrossEntropy::Eval0(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const
{
	size_t right = y(0);
	return -log(x(right));
}

Eigen::RowVectorXd CrossEntropy::Eval1(const Eigen::VectorXd &x,
                                       const Eigen::VectorXd &y) const
{
	size_t right = y(0);
	Eigen::RowVectorXd result = Eigen::RowVectorXd::Zero(x.rows());
	result(right) = -1 / x(right);

	return result;
}

}  // namespace neural::loss
