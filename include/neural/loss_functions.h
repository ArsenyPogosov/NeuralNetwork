#include <Eigen/Dense>

namespace neural::loss
{

constexpr class MSE
{
public:
	double Eval0(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const;
	Eigen::RowVectorXd Eval1(const Eigen::VectorXd &x,
	                         const Eigen::VectorXd &y) const;
} MSE;

constexpr class CrossEntropy
{
public:
	double Eval0(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const;
	Eigen::RowVectorXd Eval1(const Eigen::VectorXd &x,
	                         const Eigen::VectorXd &y) const;
} CrossEntropy;

}  // namespace neural::loss
