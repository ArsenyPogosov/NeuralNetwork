#include <loss_functions.h>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using neural::loss::CrossEntropy;

TEST_CASE("CrossEntropy is right", "[loss]")
{
	Eigen::VectorXd x(4), y(1);
	x << 0.1, 0.2, 0.3, 0.4;
	y << 3;
	double loss = 0.91629073187;
	Eigen::RowVectorXd derivative(4);
	derivative << 0, 0, 0, -2.5;

	SECTION("eval0 is right")
	{
		REQUIRE(CrossEntropy.Eval0(x, y) == Catch::Approx(loss));
	}

	SECTION("eval1 is right")
	{
		REQUIRE(CrossEntropy.Eval1(x, y).isApprox(derivative));
	}
}

