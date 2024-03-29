# seagull 1.1.0

* Added new algorithm: fitted group lasso.

* Added new algorithm: fitted sparse-group lasso.

* Added new parameter: l2_fitted_values to enable the new algorithms.

* Added new parameter: step_size - mandatory for the fitted sparse-group lasso.

* Added new parameter: delta - mandatory for both new algorithms.

* Added new parameter: standardize to enable automated standardization of input
data.

* Imported the R package "matrixStats" since it is used for the standardization.

* Updated the help pages and vignette to provide information about the new
algorithms and parameters.


# seagull 1.0.6

* Deleted "trace_progress = T" in all examples, since it caused --run-donttest
to fail.

* Added a reference in the package documentation.


# seagull 1.0.5

* Wrapped the entire example section in seagull.R into a \donttest environment
to circumvent troubles with the Solaris OS.


# seagull 1.0.4

* Replaced the pre-calculation of the matrix `X^T X`, by matrix-vector
multiplications for the gradients in src\seagull_lasso.cpp,
src\seagull_group_lasso.cpp, and src\seagull_sparse_group_lasso.cpp. This
circumvents potential memory issues for the allocation of `X^T X`.

* Fixed an issue related to the variable TEMP2 in src\seagull_lasso.cpp,
src\seagull_group_lasso.cpp, and src\seagull_sparse_group_lasso.cpp


# seagull 1.0.3

* Replaced `\dontrun` by `\donttest` in R\seagull.R.

* Shortened title in DESCRIPTION.


# seagull 1.0.2

* Fixed a limitation for the design matrix X. The matrix may now have more
columns than rows. But each default algorithm to calculate `max_lambda` will
fail, because the inverse of `X^T X` is explicitly needed. However, if a value
for `max_lambda` is provided upon calling the function `seagull`, a solution
will be calculated.


# seagull 1.0.1

* Added parameter `trace_progress`. Default is `FALSE`.

* Added general vignette.


# seagull 1.0.0

* Exchanged wrappers `seagull_lasso`, `seagull_group_lasso`, and
`seagull_sparse_group_lasso` by `seagull`. The different penalties shall now be
called by specifying the mixing parameter `alpha`. This parameter was initially
only necessary for the sparse-group lasso. But the lasso and the group lasso are
limiting cases, where `alpha = 1` and `alpha = 0`, respectively. So, now both
regularizations may be initialized by calling the function `seagull` with
`alpha = 1` or `alpha = 0`.


# seagull 0.1.2

* Added documentation.


# seagull 0.1.1

* Added different exemplary data set (`seagull_data`).


# seagull 0.1.0

* Added wrapper `seagull_lasso` for the `seagull_lasso_Rcpp.cpp`

* Added wrapper `seagull_group_lasso` for the `seagull_group_lasso_Rcpp.cpp`

* Added wrapper `seagull_sparse_group_lasso` for the
`seagull_sparse_group_lasso_Rcpp.cpp`