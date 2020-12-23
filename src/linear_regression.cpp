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

	
	Matrix Xt = X.transpose();
	W = (Xt * X).ldlt().solve(Xt * y);
	
}

Matrix LinearRegression::predict(Matrix X)
{
    //auto ret = MatrixXd::Zero(X.rows(), 1);

	return (X * W);
}

Matrix LinearRegression::coef()
{
	return W;
}
