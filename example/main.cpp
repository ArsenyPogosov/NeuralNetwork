#include <neural/activation_functions.h>
#include <neural/loss_functions.h>
#include <neural/neural_network.h>

#include <iostream>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

Matrix Foo(const Matrix &x)
{
	Matrix y(2, x.cols());
	for (int i = 0; i < x.cols(); ++i)
	{
		y(0, i) = x.col(i).minCoeff();
		y(1, i) = x.col(i).maxCoeff();
	}

	return y;
}

int main()
{
	Matrix x_train, y_train, x_test, y_test;
	x_train = Matrix::Random(3, 100);
	y_train = Foo(x_train);
	x_test = Matrix::Random(3, 100);
	y_test = Foo(x_test);

	neural::NeuralNetwork net(
	    {3, 3, 2}, {neural::activation::ReLu, neural::activation::Linear},
	    neural::loss::MSE);
	net.SetLearningRate([](int epoch) { return 0.1 / (1 + epoch * 0.01); });

	std::cout << "Training for a minute\n";
	net.Train(x_train, y_train, 0, 1000 * 60, 0);

	std::cout << "Accuracy on train: " << net.MeanLoss(x_train, y_train)
	          << '\n';
	std::cout << "Accuracy on test: " << net.MeanLoss(x_test, y_test) << '\n';

	Vector a(3);
	a << 1, 2, 3,
	    std::cout << "x = \n"
	              << a << '\n'
	              << "net(x) = \n"
	              << net.Eval(a) << '\n';

	return 0;
}

