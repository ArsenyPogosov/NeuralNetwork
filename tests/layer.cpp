#include <layer.h>

#include <catch2/catch_test_macros.hpp>

using neural::layer::Gradient;
using neural::layer::Layer;

class Dummy
{
public:
	Eigen::VectorXd Eval0(Eigen::VectorXd x)
	{
		return Eigen::VectorXd(x.rows());
	}
	Eigen::MatrixXd Eval1(Eigen::VectorXd x)
	{
		return Eigen::MatrixXd(x.rows(), x.rows());
	}
};

class Ex
{
public:
	Eigen::VectorXd Eval0(Eigen::VectorXd x)
	{
		return x.array().exp();
	}
	Eigen::MatrixXd Eval1(Eigen::VectorXd x)
	{
		return x.array().exp().matrix().asDiagonal();
	}
};

TEST_CASE("Layer CoefsCount is right", "[layer]")
{
	Layer l(3, 4, Dummy{});
	REQUIRE(l.CoefsCount() == 3 * 4 + 4);
	REQUIRE(l.CoefsCount() == l.GetCoefs().size());
}

TEST_CASE("Layer LoadCoefs is right", "[layer]")
{
	Layer l(3, 2, Dummy{});
	std::vector<double> coefs{1, 2, 3, 4, 5, 4, 3, 2};
	REQUIRE_NOTHROW(l.LoadCoefs(coefs));
	REQUIRE(l.GetCoefs() == coefs);
}

// very ugly(
TEST_CASE("Layer moves towards lower value", "[layer]")
{
	for (int i = 0; i < 5; ++i)
	{
		Layer l(3, 3, Ex{});
		Eigen::VectorXd x(3);
		x << 1, 2, 3;

		Eigen::VectorXd y1 = l.PushForward(x);
		double v1 = y1.squaredNorm();
		Eigen::RowVectorXd u = y1.transpose() * 2;

		Gradient grad = l.GetGrad(x, u);
		grad *= 0.01;
		l.Descend(grad);

		Eigen::VectorXd y2 = l.PushForward(x);
		double v2 = y2.squaredNorm();

		REQUIRE(v1 > v2);
	}
}
