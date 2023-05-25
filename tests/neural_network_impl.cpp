#include <activation_functions.h>
#include <loss_functions.h>
#include <neural_network_impl.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using std::milli;

using Network = neural::NeuralNetworkImpl;

TEST_CASE("NeuralNetwork", "[neuralnetwork]")
{
	Network net({3, 2, 2},
	            {neural::activation::ReLu, neural::activation::Linear},
	            neural::loss::MSE);

	SECTION("NeuralNetwork evalutions make sense")
	{
		Eigen::MatrixXd x(3, 5), y;
		REQUIRE_NOTHROW(y = net.Eval(x));
		REQUIRE(y.rows() == 2);
		REQUIRE(y.cols() == 5);
	}

	SECTION("Can change batch size")
	{
		REQUIRE_NOTHROW(net.SetBatchSize(200));
	}

	SECTION("Mean is right")
	{
		Eigen::MatrixXd x(3, 5), y(2, 5), z(2, 5);
		x.setRandom();
		y.setRandom();
		REQUIRE_NOTHROW(z = net.Eval(x));

		double mean_loss = 0;
		for (int i = 0; i < 5; ++i)
			mean_loss += neural::loss::MSE.Eval0(z.col(i), y.col(i));
		mean_loss /= 5;

		REQUIRE(net.MeanLoss(x, y) == Catch::Approx(mean_loss));
	}

	SECTION("Stops on time")
	{
		auto start_time = std::chrono::steady_clock::now();

		net.Train(Eigen::MatrixXd(3, 5).setRandom(),
		          Eigen::MatrixXd(2, 5).setRandom(), 0, 1000, 0);

		auto end_time = std::chrono::steady_clock::now();
		auto duration =
		    std::chrono::duration<double, milli>(end_time - start_time);
		REQUIRE(duration.count() >= 1000);
	}

	SECTION("Stops on err")
	{
		Eigen::MatrixXd x(3, 100), y(2, 100);
		x.setRandom();
		y = x.topRows(2);

		net.Train(x, y, 1, 0, 0);

		REQUIRE(net.MeanLoss(x, y) <= 1);
	}

	SECTION("Load is right")
	{
		std::vector<double> coefs;
		REQUIRE_NOTHROW(coefs = net.DumpCoefs());
		coefs[0] += 1;
		coefs.back() -= 2;
		REQUIRE_NOTHROW(net.LoadCoefs(coefs));
		REQUIRE(net.DumpCoefs() == coefs);
	}

	SECTION("Use custom learning_rate")
	{
		int cnt = 0;
		auto foo = [&cnt](int)
		{
			++cnt;
			return 0;
		};
		REQUIRE_NOTHROW(net.SetLearningRate(foo));
		REQUIRE_NOTHROW(net.Train({}, {}, 0, 0, 10));
		REQUIRE(cnt > 0);
	}
}

