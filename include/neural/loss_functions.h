#include <Eigen/Dense>

namespace neural::loss
{

constexpr class MSE
{
public:
	double Eval0(Eigen::VectorXd x, Eigen::VectorXd y) const;
	Eigen::RowVectorXd Eval1(Eigen::VectorXd x, Eigen::VectorXd y) const;
} MSE;

constexpr class CrossEntropy
{
public:
	double Eval0(Eigen::VectorXd x, Eigen::VectorXd y) const;
	Eigen::RowVectorXd Eval1(Eigen::VectorXd x, Eigen::VectorXd y) const;
} CrossEntropy;

}  // namespace neural::loss
