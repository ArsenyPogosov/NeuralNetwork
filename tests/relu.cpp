#include <activation_functions.h>

#include <catch2/catch_test_macros.hpp>

using neural::activation::ReLu;

TEST_CASE("ReLu is right", "[activation]")
{
	Eigen::VectorXd input(6);
	input << std::numeric_limits<double>::lowest(), -1, 0,
	    std::numeric_limits<double>::min(), 1,
	    std::numeric_limits<double>::max();
	Eigen::VectorXd output(6);
	output << 0, 0, 0, std::numeric_limits<double>::min(), 1,
	    std::numeric_limits<double>::max();
	Eigen::MatrixXd derivative1{
	    {0, 0, 0, 0, 0, 0},  //
	    {0, 0, 0, 0, 0, 0},  //
	    {0, 0, 0, 0, 0, 0},  // ReLu'(0) = 0, 1
	    {0, 0, 0, 1, 0, 0},  //
	    {0, 0, 0, 0, 1, 0},  //
	    {0, 0, 0, 0, 0, 1},  //
	};
	Eigen::MatrixXd derivative2{
	    {0, 0, 0, 0, 0, 0},  //
	    {0, 0, 0, 0, 0, 0},  //
	    {0, 0, 1, 0, 0, 0},  // ReLu'(0) = 0, 1
	    {0, 0, 0, 1, 0, 0},  //
	    {0, 0, 0, 0, 1, 0},  //
	    {0, 0, 0, 0, 0, 1},  //
	};

	SECTION("eval0 is right")
	{
		REQUIRE(ReLu.Eval0(input) == output);
	}

	SECTION("eval1 is right")
	{
		REQUIRE((ReLu.Eval1(input) == derivative1 ||
		         ReLu.Eval1(input) == derivative2));
	}
}
