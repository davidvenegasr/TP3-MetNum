#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression()
{
    // Inicializamos la estructura de regresor lineal para los coeficientes en el vector W
    Matrix m;
    this-> W = m;
}

void LinearRegression::fit(Matrix X, Matrix y)
{

    Matrix XT = X.transpose();
    
    //W^ =(XTX)^−1 XTy
    W = (XT*X).inverse()* XT*y;

}


Matrix LinearRegression::predict(Matrix X)
{
    auto ret = MatrixXd::Zero(X.rows(), 1);
    Matrix y = X*W;
    return ret;
}
