#include <Eigen/Dense>
#include <functional>
#include <memory>

#include "differentiable.h"

namespace neural
{

using LossFunction = differentiable::Differentiable<double, Eigen::RowVectorXd,
                                                    const Eigen::VectorXd &,
                                                    const Eigen::VectorXd &>;

using ActivationFunction =
    differentiable::Differentiable<Eigen::VectorXd, Eigen::MatrixXd,
                                   const Eigen::VectorXd &>;

class NeuralNetworkImpl;

class NeuralNetwork
{
public:
	NeuralNetwork(std::initializer_list<size_t> sizes,
	              std::initializer_list<ActivationFunction> sigmas,
	              LossFunction loss);

	void SetBatchSize(size_t size);
	void SetLearningRate(std::function<double(int)> learning_rate);

	Eigen::MatrixXd Eval(const Eigen::MatrixXd &x);
	double MeanLoss(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y);
	void Train(const Eigen::MatrixXd &x, const Eigen::MatrixXd &y,
	           double err_stop = 0, double time_stop_ms = 0,
	           int epoch_stop = 0);

	std::vector<double> DumpCoefs() const;
	void LoadCoefs(const std::vector<double> &coefs);

	~NeuralNetwork();

private:
	std::unique_ptr<NeuralNetworkImpl> pimpl_;
};

}  // namespace neural

