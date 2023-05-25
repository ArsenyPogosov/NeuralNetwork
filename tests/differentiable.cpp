#include <differentiable/differentiable.h>

#include <catch2/catch_test_macros.hpp>

using neural::differentiable::Differentiable;

struct Dummy
{
public:
	int Eval0(int)
	{
		return 0;
	}

	int Eval1(int)
	{
		return 0;
	}
};

struct Id
{
public:
	int Eval0(int x)
	{
		return x;
	}

	int Eval1(int x)
	{
		return 1;
	}
};

struct Line
{
public:
	double Eval0(double x, double y)
	{
		return x * y;
	}
	double Eval1(double x, double y)
	{
		return y;
	}
};

struct MultiType
{
public:
	double Eval0(std::string)
	{
		return 0;
	}
	bool Eval1(std::string)
	{
		return 0;
	}
};

struct MultiTypeAndArgs
{
public:
	int *Eval0(double, size_t, std::string &)
	{
		return nullptr;
	}
	long long *Eval1(double, size_t, std::string &)
	{
		return nullptr;
	}
};

struct NonCopyable
{
	NonCopyable() = default;
	NonCopyable(const NonCopyable &) = delete;
	NonCopyable &operator=(const NonCopyable &) = delete;
};

struct NonCopyableHolder
{
public:
	int Eval0(const NonCopyable &)
	{
		return 0;
	}
	int Eval1(const NonCopyable &)
	{
		return 0;
	}
};

struct CopyCounter
{
public:
	CopyCounter()
	{
	}
	CopyCounter(const CopyCounter &)
	{
		++cnt_;
	}
	CopyCounter(CopyCounter &&)
	{
	}

	int Eval0(int x)
	{
		return cnt_;
	}
	int Eval1(int x)
	{
		return cnt_;
	}

private:
	int cnt_ = 0;
};

struct Synchronizer
{
public:
	int Eval0(int x)
	{
		x_ = x;
		return x_;
	}
	int Eval1(int)
	{
		return x_;
	}

private:
	int x_ = 0;
};

TEST_CASE("Differentiable is constructable", "[differentiable]")
{
	Differentiable<int, int, int> d1(Dummy{});
	Differentiable<int, int, int> d2(Id{});
	Differentiable<double, double, double, double> d3(Line{});
	Differentiable<double, bool, std::string> d4(MultiType{});
	Differentiable<int *, long long *, double, size_t, std::string &> d5(
	    MultiTypeAndArgs{});
	Differentiable<int, int, const NonCopyable &> d6(NonCopyableHolder{});
	Differentiable<int, int, int> d7(CopyCounter{});
	Differentiable<int, int, int> d8(Synchronizer{});
}

TEST_CASE("Differentiable returns right", "[differentiable]")
{
	Differentiable<int, int, int> d1(Dummy{});
	REQUIRE(d1.Eval0(228) == 0);
	REQUIRE(d1.Eval1(228) == 0);
	Differentiable<int, int, int> d2(Id{});
	REQUIRE(d2.Eval0(228) == 228);
	REQUIRE(d2.Eval1(228) == 1);
	Differentiable<double, double, double, double> d3(Line{});
	REQUIRE(d3.Eval0(123, 3) == 369);
	REQUIRE(d3.Eval1(123, 3) == 3);
}

TEST_CASE("Differentiable copy implementation only when needed",
          "[differentiable]")
{
	Differentiable<int, int, int> d7(CopyCounter{});
	REQUIRE(d7.Eval0(0) == 0);
}

TEST_CASE("Differentiable share copy between Eval0 and Eval1",
          "[differentiable]")
{
	Differentiable<int, int, int> d8(Synchronizer{});
	REQUIRE_NOTHROW(d8.Eval0(228));
	REQUIRE(d8.Eval1(0) == 228);
}

TEST_CASE("Differentiable has value semantics", "[differentiable]")
{
	Differentiable<int, int, int> d81(Synchronizer{});
	Differentiable<int, int, int> d82 = d81;
	REQUIRE_NOTHROW(d81.Eval0(1));
	REQUIRE_NOTHROW(d82.Eval0(2));
	REQUIRE(d81.Eval1(0) == 1);
	REQUIRE(d82.Eval1(0) == 2);
}
