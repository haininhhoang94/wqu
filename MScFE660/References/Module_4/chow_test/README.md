# chow_test
Python module to calculate Chow break statistics.

The Chow test was developed by econometrician Gregory Chow in 1960 to test whether one regression or two or more regressions best fit the time series data. It actually tests whether there is a structural "break" in the dataset. More information can be found [on Wikipedia](https://en.wikipedia.org/wiki/Chow_test) and this [Statistics How To post](http://www.statisticshowto.com/chow-test/).

Here is the formula:
</br><img src='http://www.statisticshowto.com/wp-content/uploads/2016/10/chow-test-formula.png'>

Current version only supports simple linear models with a single x-variable, that have a time-axis in years as integers, and when finding breaks where k = 2.

You can find my blog post about this package [here](http://joshualoong.com/2018/01/05/Building-the-Python-chow-test-Package/).

## Installation
Clone this repository, move into the directory, and install with pip:
```
git clone https://github.com/jtloong/chow-test.git
cd chow-test
pip install .
```
In your Python code you can import it as:
```
import chow_test
```

## Usage

The function has four parameters, and be used to find either the f-value or p-value of your Chow test.
```
chow_test.f_value(y1, x1, y2, x2)
```
or

```
chow_test.p_value(y1, x1, y2, x2)
```

Here are the requirements for these parameters:

| Parameter | Requirement                                                                        |
|-----------|------------------------------------------------------------------------------------|
| y1        | An array-like variable representing y-value data before the proposed break point   |
| x1        | An array-like variable representing x-value data before the proposed break point   |
| y2        | An array-like variable representing y-value data after the proposed break point    |
| x2        | An array-like variable representing x-value data after the proposed break point    |


## Example

Checkout the tests folder to see an ipython notebook with the chow_test module in use.


## Future Ideas

* Build in functionality for complex linear models
* Enhance time comparison ability
* Ability to find more breaks in the data
