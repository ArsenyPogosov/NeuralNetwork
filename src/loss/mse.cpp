#include <loss_functions.h>

#include <Eigen/Dense>

namespace neural::loss
{

double MSE::Eval0(Eigen::VectorXd x, Eigen::VectorXd y) const
{
	return (x - y).squaredNorm() / x.rows();
}

Eigen::RowVectorXd MSE::Eval1(Eigen::VectorXd x, Eigen::VectorXd y) const
{
	return 2 * (x - y) / x.rows();
}

}  // namespace neural::loss
