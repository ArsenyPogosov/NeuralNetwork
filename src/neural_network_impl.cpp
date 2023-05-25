#include <defaults.h>
#include <neural_network_impl.h>

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

namespace neural
{

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

NeuralNetworkImpl::NeuralNetworkImpl(
    std::initializer_list<size_t> sizes,
    std::initializer_list<ActivationFunction> sigmas, LossFunction loss)
    : loss_(loss),
      batch_size_(consts::DefaultBatchSize),
      learning_rate_(
          [](int epoch) -> double
          {
	          return consts::DefaultStartingLearningRate /
	                 (1 + epoch * consts::DefaultLearningRateDecay);
          })
{
	layers_.reserve(sigmas.size());
	auto it1 = sizes.begin();
	auto it2 = sigmas.begin();
	for (; it2 != sigmas.end(); ++it1, ++it2)
		layers_.emplace_back(*it1, *(it1 + 1), *it2);
}

void NeuralNetworkImpl::SetBatchSize(size_t size)
{
	batch_size_ = size;
}

Matrix NeuralNetworkImpl::Eval(const Matrix& x)
{
	if (!x.cols())
		return {};

	Matrix prev = x;
	for (auto& layer : layers_)
	{
		Matrix cur = layer.PushForward(prev.col(0));
		cur.conservativeResize(cur.rows(), prev.cols());
		for (size_t i = 1; i < prev.cols(); ++i)
			cur.col(i) = layer.PushForward(prev.col(i));

		prev = cur;
	}

	return prev;
}

double NeuralNetworkImpl::MeanLoss(const Eigen::MatrixXd& x,
                                   const Eigen::MatrixXd& y)
{
	Matrix z = Eval(x);

	double sum_loss = 0;
	for (int i = 0; i < z.cols(); ++i)
		sum_loss += loss_.Eval0(z.col(i), y.col(i));

	return sum_loss / z.cols();
}

void NeuralNetworkImpl::Train(const Matrix& x, const Matrix& y, double err_stop,
                              double time_stop_ms, int epoch_stop)
{
	auto start = std::chrono::steady_clock::now();
	int epoch = 0;
	double err = std::numeric_limits<double>::max();

	auto stop = [&]()
	{
		if (err_stop != 0 && err < err_stop)
			return true;
		if (time_stop_ms != 0 && std::chrono::duration<double, std::milli>(
		                             std::chrono::steady_clock::now() - start)
		                                 .count() > time_stop_ms)
			return true;
		if (epoch_stop != 0 && epoch >= epoch_stop)
			return true;

		return false;
	};

	while (!stop())
	{
		double new_err = 0;
		double d = learning_rate_(epoch);
		for (size_t i = 0; i < x.cols(); i += batch_size_)
		{
			size_t cur_size = std::min(batch_size_, x.cols() - i);
			new_err += TrainBatch(x, y, i, cur_size, d);
		}

		err = new_err / x.cols();
		++epoch;
	}
}

std::vector<double> NeuralNetworkImpl::DumpCoefs() const
{
	size_t size = 0;
	for (auto& i : layers_)
		size += i.CoefsCount();

	std::vector<double> res(size);
	size_t it = 0;
	for (auto& layer : layers_)
		for (auto& i : layer.DumpCoefs())
			res[it++] = i;

	return res;
}

void NeuralNetworkImpl::LoadCoefs(const std::vector<double>& coefs)
{
	size_t it = 0;
	for (auto& layer : layers_)
	{
		size_t sz = layer.CoefsCount();
		std::vector cur(coefs.begin() + it, coefs.begin() + it + sz);
		it += sz;
		layer.LoadCoefs(cur);
	}
}

double NeuralNetworkImpl::TrainBatch(const Eigen::MatrixXd& x,
                                     const Eigen::MatrixXd& y, size_t i,
                                     size_t size, double d)
{
	double err = 0;
	std::vector<layer::Gradient> grads(layers_.size());

	for (size_t j = i; j < i + size; ++j)
	{
		std::vector<Vector> xs;
		xs.reserve(layers_.size() + 1);
		xs.push_back(x.col(j));
		for (auto& layer : layers_)
			xs.push_back(layer.PushForward(xs.back()));

		err += loss_.Eval0(xs.back(), y.col(j));
		RowVector u = loss_.Eval1(xs.back(), y.col(j));

		for (ssize_t k = layers_.size() - 1; k >= 0; --k)
		{
			grads[k] += layers_[k].GetGrad(xs[k], u);
			u = layers_[k].PushBackward(xs[k], u);
		}
	}

	for (int j = 0; j < layers_.size(); ++j)
	{
		grads[j] *= 1. / size * d;
		layers_[j].Descend(grads[j]);
	}

	return err;
}

}  // namespace neural
