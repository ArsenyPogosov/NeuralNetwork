#pragma once

#include <differentiable/differentiable.h>

#include <Eigen/Dense>
#include <vector>

namespace neural::layer
{

using ActivationFunction =
    differentiable::Differentiable<Eigen::VectorXd, Eigen::MatrixXd,
                                   Eigen::VectorXd>;
class Layer;

class Gradient
{
public:
	Gradient &operator+=(const Gradient &other);
	Gradient &operator*=(double d);

private:
	Eigen::MatrixXd da_;
	Eigen::VectorXd db_;

	friend Layer;
};

class Layer
{
public:
	Layer(size_t m, size_t n, ActivationFunction lambda);
	Eigen::VectorXd PushForward(const Eigen::VectorXd &x);
	Eigen::RowVectorXd PushBackward(const Eigen::VectorXd &x,
	                                const Eigen::RowVectorXd &u);
	Gradient GetGrad(const Eigen::VectorXd &x, const Eigen::RowVectorXd &u);

	void Descend(Gradient grad);

	size_t CoefsCount() const;
	std::vector<double> GetCoefs() const;
	void LoadCoefs(const std::vector<double> &coefs);

private:
	Eigen::MatrixXd a_;
	Eigen::VectorXd b_;
	ActivationFunction lambda_;
};

}  // namespace neural::layer
