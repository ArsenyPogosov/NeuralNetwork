#include <activation_functions.h>

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

using neural::activation::Linear;

TEST_CASE("Linear is right", "[activation]")
{
	Eigen::VectorXd input(6);
	input << std::numeric_limits<double>::lowest(), -1, 0,
	    std::numeric_limits<double>::min(), 1,
	    std::numeric_limits<double>::max();
	Eigen::VectorXd output = input;
	Eigen::MatrixXd derivative{
	    {1, 0, 0, 0, 0, 0},  //
	    {0, 1, 0, 0, 0, 0},  //
	    {0, 0, 1, 0, 0, 0},  //
	    {0, 0, 0, 1, 0, 0},  //
	    {0, 0, 0, 0, 1, 0},  //
	    {0, 0, 0, 0, 0, 1},  //
	};

	SECTION("eval0 is right")
	{
		REQUIRE(Linear.Eval0(input) == output);
	}

	SECTION("eval1 is right")
	{
		REQUIRE(Linear.Eval1(input) == derivative);
	}
}

