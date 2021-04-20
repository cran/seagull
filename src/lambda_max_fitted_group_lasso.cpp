#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

inline static double sqrt_double(double x) { return ::sqrt(x); }

using namespace Rcpp;
using namespace arma;

//' Maximal \eqn{\lambda}
//' 
//' @name lambda_max
//' 
//' @aliases lambda_max_fitted_group_lasso
//' 
//' @param DELTA numeric value, which is squared and added to the main diagonal
//' of \eqn{Z^{(l)T} Z^{(l)}} for group l, if this matrix is not invertible.
//' 
//' @param VECTOR_Y numeric vector of observations.
//' 
//' @param VECTOR_GROUPS integer vector specifying which effect (fixed and
//' random) belongs to which group.
//' 
//' @param VECTOR_WEIGHTS_FEATURES numeric vector of weights for the vectors of
//' fixed and random effects \eqn{[b^T, u^T]^T}. The entries may be permuted
//' corresponding to their group assignments.
//' 
//' @param VECTOR_WEIGHTS_GROUPS numeric vector of pre-calculated weights for
//' each group.
//' 
//' @param VECTOR_FULL_COLUMN_RANK Boolean vector, which harbors the information
//' of whether or not the group-wise parts of the filtered matrix Z, i.e.,
//' \eqn{Z^{(l)}} for each group l, have full column rank.
//' 
//' @param VECTOR_BETA numeric vector of features. At the end of this function,
//' the random effects are initialized with zero, but the fixed effects are
//' initialized via a least squares procedure.
//' 
//' @param MATRIX_X numeric design matrix relating y to fixed and random
//' effects \eqn{[X Z]}.
//' 
//' @export
// [[Rcpp::export]]
double lambda_max_fitted_group_lasso(
  double DELTA,
  arma::colvec &VECTOR_Y,
  IntegerVector VECTOR_GROUPS,
  arma::colvec &VECTOR_WEIGHTS_FEATURES,
  arma::colvec &VECTOR_WEIGHTS_GROUPS,
  LogicalVector VECTOR_FULL_COLUMN_RANK,
  arma::colvec &VECTOR_BETA,
  arma::mat &MATRIX_X) {
  
  int n                    = MATRIX_X.n_rows;
  int p                    = MATRIX_X.n_cols;
  int index_i              = 0;
  int index_j              = 0;
  int COUNTER              = 0;
  int COUNTER_GROUP_SIZE   = 0;
  int NUMBER_ZEROS_WEIGHTS = 0;
  int NUMBER_GROUPS        = max(VECTOR_GROUPS);
  double LAMBDA_MAX        = 0.0;
  double TEMP              = 0.0;
  
  //Determine the number of weights equal to zero:
  for (index_j = 0; index_j < p; index_j++) {
    if (VECTOR_WEIGHTS_FEATURES(index_j) == 0.0) {
      NUMBER_ZEROS_WEIGHTS = NUMBER_ZEROS_WEIGHTS + 1;
    }
  }
  
  IntegerVector VECTOR_INDEX_START (NUMBER_GROUPS);
  IntegerVector VECTOR_INDEX_END (NUMBER_GROUPS);
  IntegerVector VECTOR_GROUP_SIZES (NUMBER_GROUPS);
  NumericVector VECTOR_X_TRANSP_RESIDUAL_ACTIVEc (p);
  NumericVector VECTOR_L2_NORM_GROUPSc (NUMBER_GROUPS);
  NumericVector VECTOR_RESIDUAL_ACTIVEc (n);
  NumericVector VECTOR_TEMP1c(n);
  
  colvec VECTOR_X_TRANSP_RESIDUAL_ACTIVE(VECTOR_X_TRANSP_RESIDUAL_ACTIVEc.begin(), p, false);
  colvec VECTOR_L2_NORM_GROUPS(VECTOR_L2_NORM_GROUPSc.begin(), NUMBER_GROUPS, false);
  colvec VECTOR_RESIDUAL_ACTIVE(VECTOR_RESIDUAL_ACTIVEc.begin(), n, false);
  colvec VECTOR_TEMP1(VECTOR_TEMP1c.begin(), n, false);
  
  
  /*********************************************************
   **     Create a vector of group sizes, a vector of     **
   **     start indices, and a vector of end indices      **
   **     from the vector of groups. And also create a    **
   **     vector of group weights as group means from     **
   **     the vector of feature weights:                  **
   *********************************************************/
  COUNTER = 1;
  for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
    COUNTER_GROUP_SIZE = 0;
    for (index_j = 0; index_j < p; index_j++) {
      if (VECTOR_GROUPS(index_j) == (index_i + 1)) {
        COUNTER_GROUP_SIZE = COUNTER_GROUP_SIZE + 1;
        VECTOR_INDEX_START(index_i)    = index_j - COUNTER_GROUP_SIZE + 1;
        VECTOR_INDEX_END(index_i)      = index_j;
        VECTOR_GROUP_SIZES(index_i)    = COUNTER_GROUP_SIZE;
      }
    }
  }
  
  
  /*********************************************************
   **     Treatment, if unpenalized features are          **
   **     involved:                                       **
   *********************************************************/
  if (NUMBER_ZEROS_WEIGHTS > 0) {
    NumericVector VECTOR_BETA_ACTIVEc (NUMBER_ZEROS_WEIGHTS);
    NumericMatrix MATRIX_X_ACTIVEc (n, NUMBER_ZEROS_WEIGHTS);
    
    colvec VECTOR_BETA_ACTIVE(VECTOR_BETA_ACTIVEc.begin(), NUMBER_ZEROS_WEIGHTS, false);
    mat MATRIX_X_ACTIVE(MATRIX_X_ACTIVEc.begin(), n, NUMBER_ZEROS_WEIGHTS, false);
    
    
    /*******************************************************
     **     Calculations with "active" set:               **
     *******************************************************/
    //Determine the "active" set and create X_A = X_active:
    COUNTER = 0;
    for (index_j = 0; index_j < p; index_j++) {
      if (VECTOR_WEIGHTS_FEATURES(index_j) == 0.0) {
        for (index_i = 0; index_i < n; index_i++) {
          MATRIX_X_ACTIVE(index_i, COUNTER) = MATRIX_X(index_i, index_j);
        }
        COUNTER = COUNTER + 1;
      }
    }
    
    //Solve for beta_A in y = X_A * beta_A:
    VECTOR_BETA_ACTIVE = solve(MATRIX_X_ACTIVE, VECTOR_Y);
    
    //Create beta with beta_A:
    COUNTER = 0;
    for (index_j = 0; index_j < p; index_j++) {
      if (VECTOR_WEIGHTS_FEATURES(index_j) == 0.0) {
        VECTOR_BETA(index_j) = VECTOR_BETA_ACTIVE(COUNTER);
        COUNTER = COUNTER + 1;
      }
    }
    
    //Calculate res_A = y - X_A*beta_A:
    VECTOR_RESIDUAL_ACTIVE = VECTOR_Y - (MATRIX_X_ACTIVE * VECTOR_BETA_ACTIVE);
    
    
  /*********************************************************
   **     Treatment, if only penalized features are       **
   **     involved:                                       **
   *********************************************************/
  } else {
    //Calculate t(X)*y:
    VECTOR_RESIDUAL_ACTIVE = VECTOR_Y;
  }
  
  for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
  	NumericVector VECTOR_TEMP2c (VECTOR_GROUP_SIZES(index_i));
    colvec VECTOR_TEMP2(VECTOR_TEMP2c.begin(), VECTOR_GROUP_SIZES(index_i), false);
    NumericVector VECTOR_TEMP3c (VECTOR_GROUP_SIZES(index_i));
    colvec VECTOR_TEMP3(VECTOR_TEMP3c.begin(), VECTOR_GROUP_SIZES(index_i), false);
    
    //Create for group i the matrix t(X^(i)) * X^(i):
    mat MATRIX_X_GROUP_i (n, VECTOR_GROUP_SIZES(index_i));
    mat MATRIX_XTX_GROUP_INVERSE (VECTOR_GROUP_SIZES(index_i), VECTOR_GROUP_SIZES(index_i));
    MATRIX_X_GROUP_i         = MATRIX_X.submat(0, VECTOR_INDEX_START(index_i), n-1, VECTOR_INDEX_END(index_i));
    MATRIX_XTX_GROUP_INVERSE = MATRIX_X_GROUP_i.t() * MATRIX_X_GROUP_i;
    
  	//Add delta^2 to the main diagonal of t(X^(i)) * X^(i), if column rank of X^(i) is not full:
  	if (!VECTOR_FULL_COLUMN_RANK(index_i)) {
      for(index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
        MATRIX_XTX_GROUP_INVERSE(index_j, index_j) = MATRIX_XTX_GROUP_INVERSE(index_j, index_j) + DELTA * DELTA;
      }
  	}
    
    //Invert the current matrix:
    MATRIX_XTX_GROUP_INVERSE = inv(MATRIX_XTX_GROUP_INVERSE);
    
    VECTOR_TEMP2 = MATRIX_X_GROUP_i.t() * VECTOR_RESIDUAL_ACTIVE;
    VECTOR_TEMP3 = MATRIX_XTX_GROUP_INVERSE * VECTOR_TEMP2;
    VECTOR_TEMP1 = MATRIX_X_GROUP_i * VECTOR_TEMP3;
    
    //Scale t(X)*res_A in groups with n*weight, if weight>0:
    //Calculate l2-norm in groups:
    if (VECTOR_WEIGHTS_GROUPS(index_i) == 0.0) {
      VECTOR_L2_NORM_GROUPS(index_i) = 0.0;
    } else {
      TEMP = static_cast<double>(n) * VECTOR_WEIGHTS_GROUPS(index_i);
      for (index_j = 0; index_j < n; index_j++) {
        VECTOR_L2_NORM_GROUPS(index_i) = VECTOR_L2_NORM_GROUPS(index_i) + VECTOR_TEMP1(index_j) * VECTOR_TEMP1(index_j);
      }
      
      //Calculate l2-norm of VECTOR_TEMP3 and add to the current l2-norm, if column rank of X^(i) is not full:
      if (!VECTOR_FULL_COLUMN_RANK(index_i)) {
        for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
          VECTOR_L2_NORM_GROUPS(index_i) = VECTOR_L2_NORM_GROUPS(index_i) + DELTA * DELTA * VECTOR_TEMP3(index_j) * VECTOR_TEMP3(index_j);
        }
      }
      VECTOR_L2_NORM_GROUPS(index_i) = sqrt_double(VECTOR_L2_NORM_GROUPS(index_i)) / TEMP;
    }
  }
  
  //Determine lambda_max and perform numeric correction:
  for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
    if (VECTOR_L2_NORM_GROUPS(index_i) < 0.0) {
      VECTOR_L2_NORM_GROUPS(index_i) = -1.0 * VECTOR_L2_NORM_GROUPS(index_i);
    }
  }
  LAMBDA_MAX = max(VECTOR_L2_NORM_GROUPS);
  return LAMBDA_MAX*1.00001;
}
