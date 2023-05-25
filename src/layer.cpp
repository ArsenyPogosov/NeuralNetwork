#include <layer.h>

#include <random>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

namespace neural::layer
{

Gradient &Gradient::operator+=(const Gradient &other)
{
	da_ += other.da_;
	db_ += other.db_;

	return *this;
}

Gradient &Gradient::operator*=(double d)
{
	da_ *= d;
	db_ *= d;

	return *this;
}

Layer::Layer(size_t m, size_t n, ActivationFunction lambda) : lambda_(lambda)
{
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<double> d{0, 1};

	a_ = Matrix::NullaryExpr(n, m, [&gen, &d]() -> double { return d(gen); });
	b_ = Vector::NullaryExpr(n, [&gen, &d]() -> double { return d(gen); });
}

Vector Layer::PushForward(const Vector &x)
{
	return lambda_.Eval0(a_ * x + b_);
}

RowVector Layer::PushBackward(const Vector &x, const RowVector &u)
{
	Matrix d_lambda = lambda_.Eval1(a_ * x + b_);
	return u * d_lambda * a_;
}

Gradient Layer::GetGrad(const Vector &x, const RowVector &u)
{
	Gradient res;
	Matrix d_lambda = lambda_.Eval1(a_ * x + b_);
	res.da_ = d_lambda.transpose() * u.transpose() * x.transpose();
	res.db_ = d_lambda.transpose() * u.transpose();
	return res;
}

void Layer::Descend(Gradient grad)
{
	a_ -= grad.da_;
	b_ -= grad.db_;
}

size_t Layer::CoefsCount() const
{
	return a_.size() + b_.size();
}

std::vector<double> Layer::GetCoefs() const
{
	std::vector<double> coefs;
	coefs.reserve(CoefsCount());
	for (size_t i = 0; i < a_.rows(); ++i)
		for (size_t j = 0; j < a_.cols(); ++j)
			coefs.push_back(a_(i, j));
	for (size_t i = 0; i < b_.rows(); ++i)
		coefs.push_back(b_(i));

	return coefs;
}

void Layer::LoadCoefs(const std::vector<double> &coefs)
{
	auto coef = coefs.begin();
	for (size_t i = 0; i < a_.rows(); ++i)
		for (size_t j = 0; j < a_.cols(); ++j)
			a_(i, j) = *coef++;
	for (size_t i = 0; i < b_.rows(); ++i)
		b_(i) = *coef++;
}

}  // namespace neural::layer
