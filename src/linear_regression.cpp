#include <algorithm>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression() {
}

void LinearRegression::fit(Matrix X, Matrix y) {

	W = (X.transpose() * X).ldlt().solve(X.transpose() * y);
	
}

Matrix LinearRegression::predict(Matrix X)
{

	return (X * W);
}


