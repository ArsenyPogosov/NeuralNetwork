#pragma once

#include <differentiable/differentiable.h>
#include <layer.h>

#include <functional>
#include <initializer_list>

namespace neural
{

using LossFunction =
    differentiable::Differentiable<double, Eigen::RowVectorXd, Eigen::VectorXd,
                                   Eigen::VectorXd>;

using layer::ActivationFunction;

class NeuralNetworkImpl
{
public:
	NeuralNetworkImpl(std::initializer_list<size_t> sizes,
	                  std::initializer_list<ActivationFunction> sigmas,
	                  LossFunction loss);

	void SetBatchSize(size_t size);
	template <typename T>
	void SetLearningRate(T learning_rate)
	{
		learning_rate_ = learning_rate;
	}

	Eigen::MatrixXd Eval(const Eigen::MatrixXd &x);
	double MeanLoss(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y);
	void Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
	           double err_stop = 0, double time_stop_ms = 0,
	           int epoch_stop = 0);

	std::vector<double> DumpCoefs() const;
	void LoadCoefs(const std::vector<double> &coefs);

private:
	double TrainBatch(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
	                  size_t i, size_t size, double d);

	std::vector<layer::Layer> layers_;
	LossFunction loss_;

	size_t batch_size_;
	std::function<double(int)> learning_rate_;
};

}  // namespace neural
