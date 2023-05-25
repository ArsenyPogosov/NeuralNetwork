#pragma once

#include <any>
#include <functional>
#include <utility>

namespace neural::differentiable
{

template <typename Output, typename Derivative, typename... Inputs>
class Differentiable
{
public:
	template <typename T>
	Differentiable(T f)
	    : underlying_(std::move(f)),
	      eval0_([](std::any &f, Inputs... xn) -> Output
	             { return std::any_cast<T &>(f).Eval0(xn...); }),
	      eval1_([](std::any &f, Inputs... xn) -> Derivative
	             { return std::any_cast<T &>(f).Eval1(xn...); })
	{
	}
	Output Eval0(Inputs... xn)
	{
		return eval0_(underlying_, xn...);
	}
	Derivative Eval1(Inputs... xn)
	{
		return eval1_(underlying_, xn...);
	}

private:
	std::any underlying_;
	const std::function<Output(std::any &, Inputs...)> eval0_;
	const std::function<Derivative(std::any &, Inputs...)> eval1_;
};

}  // namespace neural::differentiable
