#include <Eigen/Dense>

namespace neural::activation
{

constexpr class Linear
{
public:
	Eigen::VectorXd Eval0(const Eigen::VectorXd &x) const;
	Eigen::MatrixXd Eval1(const Eigen::VectorXd &x) const;
} Linear;

constexpr class ReLu
{
public:
	Eigen::VectorXd Eval0(const Eigen::VectorXd &x) const;
	Eigen::MatrixXd Eval1(const Eigen::VectorXd &x) const;
} ReLu;

constexpr class Sigmoid
{
public:
	Eigen::VectorXd Eval0(const Eigen::VectorXd &x) const;
	Eigen::MatrixXd Eval1(const Eigen::VectorXd &x) const;
} Sigmoid;

constexpr class SoftMax
{
public:
	Eigen::VectorXd Eval0(const Eigen::VectorXd &x) const;
	Eigen::MatrixXd Eval1(const Eigen::VectorXd &x) const;
} SoftMax;

}  // namespace neural::activation
