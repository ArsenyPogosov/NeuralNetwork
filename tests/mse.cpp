#include <loss_functions.h>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using neural::loss::MSE;

TEST_CASE("MSE is right", "[loss]")
{
	Eigen::VectorXd x(5), y(5);
	x << -2, -1, 0, 1, 2;
	y << 0, 1, 2, 1, 0;
	double loss = 3.2;
	Eigen::RowVectorXd derivative(5);
	derivative << -0.8, -0.8, -0.8, 0, 0.8;

	SECTION("eval0 is right")
	{
		REQUIRE(MSE.Eval0(x, y) == Catch::Approx(loss));
	}

	SECTION("eval1 is right")
	{
		REQUIRE(MSE.Eval1(x, y).isApprox(derivative));
	}
}

