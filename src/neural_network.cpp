#include <differentiable.h>
#include <neural_network.h>
#include <neural_network_impl.h>

#include <Eigen/Dense>
#include <functional>
#include <memory>

namespace neural
{

NeuralNetwork::NeuralNetwork(std::initializer_list<size_t> sizes,
                             std::initializer_list<ActivationFunction> sigmas,
                             LossFunction loss)
    : pimpl_(std::make_unique<NeuralNetworkImpl>(sizes, sigmas, loss))
{
}

void NeuralNetwork::SetBatchSize(size_t size)
{
	pimpl_->SetBatchSize(size);
}
void NeuralNetwork::SetLearningRate(std::function<double(int)> learning_rate)
{
	pimpl_->SetLearningRate(learning_rate);
}

Eigen::MatrixXd NeuralNetwork::Eval(const Eigen::MatrixXd &x)
{
	return pimpl_->Eval(x);
}
double NeuralNetwork::MeanLoss(const Eigen::MatrixXd &x,
                               const Eigen::MatrixXd &y)
{
	return pimpl_->MeanLoss(x, y);
}
void NeuralNetwork::Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
                          double err_stop, double time_stop_ms, int epoch_stop)
{
	pimpl_->Train(x, y, err_stop, time_stop_ms, epoch_stop);
}

std::vector<double> NeuralNetwork::DumpCoefs() const
{
	return pimpl_->DumpCoefs();
}

void NeuralNetwork::LoadCoefs(const std::vector<double> &coefs)
{
	pimpl_->LoadCoefs(coefs);
}

}  // namespace neural
