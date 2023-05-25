#include <activation_functions.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using neural::activation::Sigmoid;

TEST_CASE("Sigmoid is right", "[activation]")
{
	Eigen::VectorXd input(6);
	input << std::numeric_limits<double>::lowest(), -1, 0,
	    std::numeric_limits<double>::min(), 1,
	    std::numeric_limits<double>::max();
	Eigen::VectorXd output(6);
	output << 0, 0.26894142137, 0.5, 0.5, 0.73105857863, 1;
	Eigen::MatrixXd derivative{
	    {0, 0, 0, 0, 0, 0},                               //
	    {0, 0.1966119332414818525374247335, 0, 0, 0, 0},  //
	    {0, 0, 0.25, 0, 0, 0},                            //
	    {0, 0, 0, 0.25, 0, 0},                            //
	    {0, 0, 0, 0, 0.1966119332414818525374247335, 0},  //
	    {0, 0, 0, 0, 0, 0},                               //
	};

	SECTION("eval0 is right")
	{
		REQUIRE(Sigmoid.Eval0(input).isApprox(output));
	}

	SECTION("eval1 is right")
	{
		REQUIRE(Sigmoid.Eval1(input).isApprox(derivative));
	}
}

