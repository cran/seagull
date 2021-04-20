#include <RcppArmadillo.h>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

inline static double sqrt_double(double x) { return ::sqrt(x); }

using namespace Rcpp;
using namespace arma;

//' Lasso, (fitted) group lasso, and (fitted) sparse-group lasso
//' 
//' @name lasso_variants
//' 
//' @aliases fitted_group_lasso
//' 
//' @param VECTOR_Yc numeric vector of observations.
//' 
//' @param Y_MEAN arithmetic mean of VECTOR_Yc.
//' 
//' @param MATRIX_Xc numeric design matrix relating y to fixed and random
//' effects \eqn{[X Z]}. The columns may be permuted corresponding to their
//' group assignments.
//' 
//' @param VECTOR_Xc_MEANS numeric vector of arithmetic means of each column
//' of MATRIX_Xc.
//' 
//' @param VECTOR_Xc_STANDARD_DEVIATIONS numeric vector of estimates of
//' standard deviations of each column of MATRIX_Xc. Values are calculated via
//' the function \code{colSds} from the R-package \code{matrixStats}.
//' 
//' @param VECTOR_WEIGHTS_FEATURESc numeric vector of weights for the vectors
//' of fixed and random effects \eqn{[b^T, u^T]^T}. The entries may be permuted
//' corresponding to their group assignments.
//' 
//' @param VECTOR_WEIGHTS_GROUPSc numeric vector of pre-calculated weights for
//' each group.
//' 
//' @param VECTOR_FULL_COLUMN_RANK Boolean vector, which harbors the information
//' of whether or not the group-wise parts of the filtered matrix Z, i.e.,
//' \eqn{Z^{(l)}} for each group l, have full column rank.
//' 
//' @param VECTOR_GROUPS integer vector specifying which effect (fixed and
//' random) belongs to which group.
//' 
//' @param VECTOR_BETAc numeric vector whose partitions will be returned
//' (partition 1: estimates of fixed effects, partition 2: predictions of random
//' effects). During the computation the entries may be in permuted order. But
//' they will be returned according to the order of the user's input.
//' 
//' @param VECTOR_INDEX_PERMUTATION integer vector that contains information
//' about the original order of the user's input.
//' 
//' @param VECTOR_INDEX_EXCLUDE integer vector, which contains the indices of
//' every column that was filtered due to low standard deviation. This vector
//' only has an effect, if \code{standardize = TRUE} is used.
//' 
//' @param EPSILON_CONVERGENCE value for relative accuracy of the solution to
//' stop the algorithm for the current value of \eqn{\lambda}. The algorithm
//' stops after iteration m, if: \deqn{||sol^{(m)} - sol^{(m-1)}||_\infty <
//' \epsilon_c * ||sol1{(m-1)}||_2.}
//' 
//' @param ITERATION_MAX maximum number of iterations for each value of the
//' penalty parameter \eqn{\lambda}. Determines the end of the calculation if
//' the algorithm didn't converge according to \code{EPSILON_CONVERGENCE}
//' before.
//' 
//' @param GAMMA multiplicative parameter to decrease the step size during
//' backtracking line search. Has to satisfy: \eqn{0 < \gamma < 1}.
//' 
//' @param LAMBDA_MAX maximum value for the penalty parameter. This is the start
//' value for the grid search of the penalty parameter \eqn{\lambda}.
//' 
//' @param PROPORTION_XI multiplicative parameter to determine the minimum value
//' of \eqn{\lambda} for the grid search, i.e. \eqn{\lambda_{min} = \xi *
//' \lambda_{max}}. Has to satisfy: \eqn{0 < \xi \le 1}. If \code{xi=1}, only a
//' single solution for \eqn{\lambda = \lambda_{max}} is calculated.
//' 
//' @param DELTA numeric value, which is squared and added to the main diagonal
//' of \eqn{Z^{(l)T} Z^{(l)}} for group l, if this matrix is not invertible.
//' 
//' @param NUMBER_INTERVALS number of lambdas for the grid search between
//' \eqn{\lambda_{max}} and \eqn{\xi * \lambda_{max}}. Loops are performed on a 
//' logarithmic grid.
//' 
//' @param NUMBER_FIXED_EFFECTS non-negative integer to determine the number of
//' fixed effects present in the mixed model.
//' 
//' @param NUMBER_VARIABLES non-negative integer which corresponds to the sum
//' of all columns of the initial model matrices X and Z.
//' 
//' @param INTERNAL_STANDARDIZATION if \code{TRUE}, the input vector y is
//' centered, and each column of the input matrices X and Z is centered and
//' scaled with an internal process. Additionally, a filter is applied to X and
//' Z, which filters columns with standard deviation less than \code{1.e-7}.
//' 
//' @param TRACE_PROGRESS if \code{TRUE}, a message will occur on the screen
//' after each finished loop of the \eqn{\lambda} grid. This is particularly
//' useful for larger data sets.
//' 
//' @export
// [[Rcpp::export]]
List seagull_fitted_group_lasso(
  NumericVector VECTOR_Yc,
  double Y_MEAN,
  NumericMatrix MATRIX_Xc,
  NumericVector VECTOR_Xc_MEANS,
  NumericVector VECTOR_Xc_STANDARD_DEVIATIONS,
  NumericVector VECTOR_WEIGHTS_FEATURESc,
  NumericVector VECTOR_WEIGHTS_GROUPSc,
  LogicalVector VECTOR_FULL_COLUMN_RANK,
  IntegerVector VECTOR_GROUPS,
  NumericVector VECTOR_BETAc,
  IntegerVector VECTOR_INDEX_PERMUTATION,
  IntegerVector VECTOR_INDEX_EXCLUDE,
  double EPSILON_CONVERGENCE,
  int ITERATION_MAX,
  double GAMMA,
  double LAMBDA_MAX,
  double PROPORTION_XI,
  double DELTA,
  int NUMBER_INTERVALS,
  int NUMBER_FIXED_EFFECTS,
  int NUMBER_VARIABLES,
  bool INTERNAL_STANDARDIZATION,
  bool TRACE_PROGRESS
  ) {
  
  
  /*********************************************************
   **     First initialization based on input variables:  **
   *********************************************************/
  int n             = VECTOR_Yc.size();
  int p             = VECTOR_WEIGHTS_FEATURESc.size();
  int NUMBER_GROUPS = max(VECTOR_GROUPS);
  colvec VECTOR_Y(VECTOR_Yc.begin(), n, false);
  colvec VECTOR_X_MEANS(VECTOR_Xc_MEANS.begin(), p, false);
  colvec VECTOR_X_STANDARD_DEVIATIONS(VECTOR_Xc_STANDARD_DEVIATIONS.begin(), p, false);
  colvec VECTOR_WEIGHTS_FEATURES(VECTOR_WEIGHTS_FEATURESc.begin(), p, false);
  colvec VECTOR_WEIGHTS_GROUPS(VECTOR_WEIGHTS_GROUPSc.begin(), NUMBER_GROUPS, false);
  colvec VECTOR_BETA(VECTOR_BETAc.begin(), p, false);
  mat MATRIX_X(MATRIX_Xc.begin(), n, p, false);
  
  
  /*********************************************************
   **     Declaration and initialization of new internal  **
   **     variables:                                      **
   *********************************************************/
  int index_i                 = 0;
  int index_j                 = 0;
  int index_interval          = 0;
  int COUNTER_GROUP_SIZE      = 0;
  int COUNTER                 = 0;
  int INDEX_EXCLUDE           = 0;
  double LAMBDA               = 0.0;
  double STEP_SIZE            = 0.0;
  double TEMP1                = 0.0;
  double TEMP2                = 0.0;
  double TEMP3                = 0.0;
  double TEMP4                = 0.0;
  double TEMP5                = 0.0;
  double THETA                = 0.0;
  double THETA_NEW            = 0.0;
  double L2_NORM_VECTOR_TEMP1 = 0.0;
  double L2_NORM_VECTOR_TEMP3 = 0.0;
  double SCALING              = 0.0;
  bool ACCURACY_REACHED       = false;
  bool CRITERION_FULFILLED    = false;
  
  IntegerVector VECTOR_INDEX_START (NUMBER_GROUPS);
  IntegerVector VECTOR_INDEX_END (NUMBER_GROUPS);
  IntegerVector VECTOR_GROUP_SIZES (NUMBER_GROUPS);
  NumericVector VECTOR_BETA_NEWc (p);
  NumericVector VECTOR_ETAc (p);
  NumericVector VECTOR_GRADIENTc (p);
  NumericVector VECTOR_X_TRANSP_Yc (p);
  NumericVector VECTOR_TEMP3c (n);
  NumericVector VECTOR_TEMP4c (n);
  NumericVector VECTOR_TEMP5c (n);
  NumericVector VECTOR_TEMP6c (p);
  NumericVector VECTOR_TEMP_GRADIENTc (n);
  
  colvec VECTOR_BETA_NEW(VECTOR_BETA_NEWc.begin(), p, false);
  colvec VECTOR_ETA(VECTOR_ETAc.begin(), p, false);
  colvec VECTOR_GRADIENT(VECTOR_GRADIENTc.begin(), p, false);
  colvec VECTOR_X_TRANSP_Y(VECTOR_X_TRANSP_Yc.begin(), p, false);
  colvec VECTOR_TEMP3(VECTOR_TEMP3c.begin(), n, false);
  colvec VECTOR_TEMP4(VECTOR_TEMP4c.begin(), n, false);
  colvec VECTOR_TEMP5(VECTOR_TEMP5c.begin(), n, false);
  colvec VECTOR_TEMP6(VECTOR_TEMP6c.begin(), p, false);
  colvec VECTOR_TEMP_GRADIENT(VECTOR_TEMP_GRADIENTc.begin(), n, false);
  
  //Additional output variables:
  IntegerVector VECTOR_ITERATIONS (NUMBER_INTERVALS);
  NumericVector VECTOR_LAMBDA (NUMBER_INTERVALS);
  NumericVector VECTOR_INTERCEPT (NUMBER_INTERVALS);
  NumericMatrix MATRIX_SOLUTION (NUMBER_INTERVALS, NUMBER_VARIABLES);
  
  
  /*********************************************************
   **     Create a vector of group sizes, a vector of     **
   **     start indices, and a vector of end indices      **
   **     from the vector of groups. And also create a    **
   **     vector of group weights as group means from     **
   **     the vector of feature weights:                  **
   *********************************************************/
  for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
    COUNTER_GROUP_SIZE = 0;
    for (index_j = 0; index_j < p; index_j++) {
      if (VECTOR_GROUPS(index_j) == (index_i + 1)) {
        COUNTER_GROUP_SIZE          = COUNTER_GROUP_SIZE + 1;
        VECTOR_INDEX_START(index_i) = index_j - COUNTER_GROUP_SIZE + 1;
        VECTOR_INDEX_END(index_i)   = index_j;
        VECTOR_GROUP_SIZES(index_i) = COUNTER_GROUP_SIZE;
      }
    }
  }
  
  
  /*********************************************************
   **     Beginning of proximal gradient descent:         **
   *********************************************************/
  //Calculate t(X)*y:
  VECTOR_X_TRANSP_Y = MATRIX_X.t() * VECTOR_Y;
  
  for (index_interval = 0; index_interval < NUMBER_INTERVALS; index_interval++) {
    Rcpp::checkUserInterrupt();
    ACCURACY_REACHED = false;
    COUNTER          = 1;
    VECTOR_ETA       = VECTOR_BETA;
    THETA            = 1.0;
    if (NUMBER_INTERVALS > 1) {
      LAMBDA = LAMBDA_MAX * exp((static_cast<double>(index_interval) / static_cast<double>(NUMBER_INTERVALS - 1)) * log(PROPORTION_XI));
    } else {
      LAMBDA = LAMBDA_MAX;
    }
    
    while ((!ACCURACY_REACHED) && (COUNTER <= ITERATION_MAX)) {
      //Calculate unscaled gradient t(X)*X*beta - t(X)*y:
      VECTOR_TEMP_GRADIENT = MATRIX_X * VECTOR_BETA;
      VECTOR_GRADIENT      = MATRIX_X.t() * VECTOR_TEMP_GRADIENT;
      VECTOR_GRADIENT      = VECTOR_GRADIENT - VECTOR_X_TRANSP_Y;
      
      //Add delta^2 * beta^(i), if column rank of X^(i) is not full:
      for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
        if (!VECTOR_FULL_COLUMN_RANK(index_i)) {
          for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
          	VECTOR_GRADIENT(VECTOR_INDEX_START(index_i) + index_j) = VECTOR_GRADIENT(VECTOR_INDEX_START(index_i) + index_j) + 
          	  (DELTA * DELTA * VECTOR_BETA(VECTOR_INDEX_START(index_i) + index_j));
          }
        }
      }
      
      //Scale gradient with n:
      for (index_j = 0; index_j < p; index_j++) {
        VECTOR_GRADIENT(index_j) = VECTOR_GRADIENT(index_j) / static_cast<double>(n);
      }
      CRITERION_FULFILLED = false;
      STEP_SIZE = 1.0;
      
      
      /*****************************************************
       **     Backtracking line search:                   **
       *****************************************************/
      while (!CRITERION_FULFILLED) {
        //Preparation for soft-scaling:
        for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
          NumericVector VECTOR_TEMP1c (VECTOR_GROUP_SIZES(index_i));
          colvec VECTOR_TEMP1(VECTOR_TEMP1c.begin(), VECTOR_GROUP_SIZES(index_i), false);
          NumericVector VECTOR_TEMP2c (VECTOR_GROUP_SIZES(index_i));
          colvec VECTOR_TEMP2(VECTOR_TEMP2c.begin(), VECTOR_GROUP_SIZES(index_i), false);
          
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
          
          //Calculate "matrix_above" * gradient^(i):
          colvec VECTOR_GRADIENT_GROUP_i(VECTOR_GROUP_SIZES(index_i));
          VECTOR_GRADIENT_GROUP_i = VECTOR_GRADIENT.subvec(VECTOR_INDEX_START(index_i), VECTOR_INDEX_END(index_i));
          VECTOR_GRADIENT_GROUP_i = MATRIX_XTX_GROUP_INVERSE * VECTOR_GRADIENT_GROUP_i;
          
          //Calculate beta^(i) - t * "vector_above":
          for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
            VECTOR_TEMP1(index_j) = VECTOR_BETA(VECTOR_INDEX_START(index_i) + index_j) - 
              STEP_SIZE * VECTOR_GRADIENT_GROUP_i(index_j);
          }
          TEMP1        = STEP_SIZE * LAMBDA * VECTOR_WEIGHTS_GROUPS(index_i);
          VECTOR_TEMP3 = MATRIX_X_GROUP_i * VECTOR_TEMP1;
          
          //Soft-scaling in groups to obtain beta_new:
          L2_NORM_VECTOR_TEMP1 = 0.0;
          L2_NORM_VECTOR_TEMP3 = 0.0;
          for (index_j = 0; index_j < n; index_j++) {
            L2_NORM_VECTOR_TEMP3 = L2_NORM_VECTOR_TEMP3 + VECTOR_TEMP3(index_j) * VECTOR_TEMP3(index_j);
          }
          
          //Calculate l2-norm of VECTOR_TEMP3 and add to the current l2-norm, if column rank of X^(i) is not full:
          if (!VECTOR_FULL_COLUMN_RANK(index_i)) {
            for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
              L2_NORM_VECTOR_TEMP1 = L2_NORM_VECTOR_TEMP1 + VECTOR_TEMP1(index_j) * VECTOR_TEMP1(index_j);
            }
            L2_NORM_VECTOR_TEMP3 = L2_NORM_VECTOR_TEMP3 + DELTA * DELTA * L2_NORM_VECTOR_TEMP1;
          }
          L2_NORM_VECTOR_TEMP3 = sqrt_double(L2_NORM_VECTOR_TEMP3);
          
          if (L2_NORM_VECTOR_TEMP3 > TEMP1) {
            SCALING = 1.0 - TEMP1 / L2_NORM_VECTOR_TEMP3;
            for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
              VECTOR_BETA_NEW(VECTOR_INDEX_START(index_i) + index_j) = SCALING * VECTOR_TEMP1(index_j);
            }
          } else {
            for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
              VECTOR_BETA_NEW(VECTOR_INDEX_START(index_i) + index_j) = 0.0;
            }
          }
        }
        
        //Reset:
        TEMP1 = 0.0;
        TEMP2 = 0.0;
        TEMP3 = 0.0;
        TEMP4 = 0.0;
        TEMP5 = 0.0;
        
        //(X * beta^(m-1)), (X * beta^(m)), (X * beta^(m-1) - X * beta^(m)), and (beta^(m-1) - beta^(m)):
        VECTOR_TEMP3 = MATRIX_X * VECTOR_BETA;
        VECTOR_TEMP4 = MATRIX_X * VECTOR_BETA_NEW;
        VECTOR_TEMP5 = VECTOR_TEMP3 - VECTOR_TEMP4;
        VECTOR_TEMP6 = VECTOR_BETA - VECTOR_BETA_NEW;
        
        //(y - X * beta^(m-1)) and (y - X * beta^(m)):
        VECTOR_TEMP3 = VECTOR_Y - VECTOR_TEMP3;
        VECTOR_TEMP4 = VECTOR_Y - VECTOR_TEMP4;
        
        //Squared l2-norm of (y - X * beta^(m-1)), (y - X * beta^(m)), (beta^(m-1) - beta^(m)),
        //and scalar product of (y - X * beta^(m-1)) and (X * beta^(m-1) - X * beta^(m)):
        TEMP1 = as_scalar(VECTOR_TEMP3.t() * VECTOR_TEMP3);
        TEMP2 = as_scalar(VECTOR_TEMP4.t() * VECTOR_TEMP4);
        TEMP3 = as_scalar(VECTOR_TEMP6.t() * VECTOR_TEMP6);
        TEMP4 = as_scalar(VECTOR_TEMP3.t() * VECTOR_TEMP5);
        
        //Sum_i delta^2 * || beta_i^(m-1) - beta_i^(m) ||_2^2:
        for (index_i = 0; index_i < NUMBER_GROUPS; index_i++) {
          if (!VECTOR_FULL_COLUMN_RANK(index_i)) {
            for (index_j = 0; index_j < VECTOR_GROUP_SIZES(index_i); index_j++) {
          	  TEMP5 = TEMP5 + DELTA * DELTA * VECTOR_TEMP6(VECTOR_INDEX_START(index_i) + index_j) * VECTOR_TEMP6(VECTOR_INDEX_START(index_i) + index_j);
            }
          }
        }
        
        //Scale the temporary variables properly:
        TEMP3 = static_cast<double>(n) * TEMP3 / STEP_SIZE;
        TEMP4 = 2.0 * TEMP4;
        
        //Full right-hand side for backtracking:
        //TEMP1 = TEMP1 + TEMP3 + TEMP4 - TEMP5;
        TEMP1 = TEMP1 + TEMP3 + TEMP4;
        
        //Decrease time step t by a factor of gamma, if necessary:
        if (TEMP2 > TEMP1) {
          STEP_SIZE = STEP_SIZE * GAMMA;
          
        //Check for convergence, if time step size is alright:
        } else {
          //l_inf_norm(beta-beta_new):
          for (index_j = 0; index_j < p; index_j++) {
            if (VECTOR_TEMP6(index_j) < 0.0) {
              VECTOR_TEMP6(index_j) = -1.0 * VECTOR_TEMP6(index_j);
            }
          }
          TEMP1 = max(VECTOR_TEMP6);
          TEMP2 = 0.0;
          
          //l2_norm(beta) * epsilon_conv:
          TEMP2 = as_scalar(VECTOR_BETA.t() * VECTOR_BETA);
          TEMP2 = sqrt_double(TEMP2) * EPSILON_CONVERGENCE;
          
          if (TEMP1 <= TEMP2) {
            ACCURACY_REACHED = true;
          }
          
          //Update: beta=beta_new:
          THETA_NEW = 0.5 * (1.0 + sqrt(1.0 + 4.0 * THETA * THETA));
          TEMP1     = (THETA - 1.0) / THETA_NEW;
          THETA     = THETA_NEW;
          
          VECTOR_BETA = VECTOR_BETA_NEW + TEMP1 * (VECTOR_BETA_NEW - VECTOR_ETA);
          VECTOR_ETA  = VECTOR_BETA_NEW;
          
          CRITERION_FULFILLED = true;
        }
      }
      
      COUNTER = COUNTER + 1;
    }
    if (TRACE_PROGRESS) {
      Rcout << "Loop " << index_interval + 1 << " of " << NUMBER_INTERVALS << " finished." << std::endl;
    }
    
    //Store solution as single row in a matrix:
    if (VECTOR_INDEX_EXCLUDE(0) == -1) {
      for (index_j = 0; index_j < NUMBER_VARIABLES; index_j++) {
        MATRIX_SOLUTION(index_interval, VECTOR_INDEX_PERMUTATION(index_j) - 1) = VECTOR_BETA(index_j) / VECTOR_X_STANDARD_DEVIATIONS(index_j);
      }
    } else {
      INDEX_EXCLUDE = 0;
    	for (index_j = 0; index_j < NUMBER_VARIABLES; index_j++) {
    	  if ((INDEX_EXCLUDE < VECTOR_INDEX_EXCLUDE.size()) && (index_j == VECTOR_INDEX_EXCLUDE(INDEX_EXCLUDE) - 1)) {
    	  	MATRIX_SOLUTION(index_interval, index_j) = 0.0;
    	  	INDEX_EXCLUDE = INDEX_EXCLUDE + 1;
    	  } else {
    	  	MATRIX_SOLUTION(index_interval, VECTOR_INDEX_PERMUTATION(index_j) - 1) = VECTOR_BETA(index_j - INDEX_EXCLUDE) / VECTOR_X_STANDARD_DEVIATIONS(index_j - INDEX_EXCLUDE);
    	  }
      }
    }
    
    //Store information about iterations and lambda in a vector:
    VECTOR_ITERATIONS(index_interval) = COUNTER - 1;
    VECTOR_LAMBDA(index_interval) = LAMBDA;
    
    VECTOR_INTERCEPT(index_interval) = Y_MEAN;
    for (index_j = 0; index_j < p; index_j++) {
    	VECTOR_INTERCEPT(index_interval) = VECTOR_INTERCEPT(index_interval) - VECTOR_X_MEANS(index_j) * VECTOR_BETA(index_j) / VECTOR_X_STANDARD_DEVIATIONS(index_j);
    }
  }
  
  
  /*********************************************************
   **     Prepare results as list and return list:        **
   *********************************************************/
  if (INTERNAL_STANDARDIZATION) {
    if (NUMBER_FIXED_EFFECTS == 0) {
      return List::create(Named("intercept")      = VECTOR_INTERCEPT,
    	                    Named("random_effects") = MATRIX_SOLUTION,
                          Named("lambda")         = VECTOR_LAMBDA,
                          Named("iterations")     = VECTOR_ITERATIONS,
                          Named("rel_acc")        = EPSILON_CONVERGENCE,
                          Named("max_iter")       = ITERATION_MAX,
                          Named("gamma_bls")      = GAMMA,
                          Named("xi")             = PROPORTION_XI,
                          Named("loops_lambda")   = NUMBER_INTERVALS);
    } else {
      NumericMatrix MATRIX_SOLUTION_FIXED (NUMBER_INTERVALS, NUMBER_FIXED_EFFECTS);
      NumericMatrix MATRIX_SOLUTION_RANDOM (NUMBER_INTERVALS, (NUMBER_VARIABLES - NUMBER_FIXED_EFFECTS));
      
      for (index_i = 0; index_i < NUMBER_INTERVALS; index_i++) {
        for (index_j = 0; index_j < NUMBER_FIXED_EFFECTS; index_j++) {
          MATRIX_SOLUTION_FIXED(index_i, index_j) = MATRIX_SOLUTION(index_i, index_j);
        }
        for (index_j = 0; index_j < (NUMBER_VARIABLES - NUMBER_FIXED_EFFECTS); index_j++) {
          MATRIX_SOLUTION_RANDOM(index_i, index_j) = MATRIX_SOLUTION(index_i, NUMBER_FIXED_EFFECTS + index_j);
        }
      }
      
      return List::create(Named("intercept")      = VECTOR_INTERCEPT,
    	                    Named("fixed_effects")  = MATRIX_SOLUTION_FIXED,
                          Named("random_effects") = MATRIX_SOLUTION_RANDOM,
                          Named("lambda")         = VECTOR_LAMBDA,
                          Named("iterations")     = VECTOR_ITERATIONS,
                          Named("rel_acc")        = EPSILON_CONVERGENCE,
                          Named("max_iter")       = ITERATION_MAX,
                          Named("gamma_bls")      = GAMMA,
                          Named("xi")             = PROPORTION_XI,
                          Named("loops_lambda")   = NUMBER_INTERVALS);
    }
  } else {
  	if (NUMBER_FIXED_EFFECTS == 0) {
      return List::create(Named("random_effects") = MATRIX_SOLUTION,
                          Named("lambda")         = VECTOR_LAMBDA,
                          Named("iterations")     = VECTOR_ITERATIONS,
                          Named("rel_acc")        = EPSILON_CONVERGENCE,
                          Named("max_iter")       = ITERATION_MAX,
                          Named("gamma_bls")      = GAMMA,
                          Named("xi")             = PROPORTION_XI,
                          Named("loops_lambda")   = NUMBER_INTERVALS);
    } else {
      NumericMatrix MATRIX_SOLUTION_FIXED (NUMBER_INTERVALS, NUMBER_FIXED_EFFECTS);
      NumericMatrix MATRIX_SOLUTION_RANDOM (NUMBER_INTERVALS, (NUMBER_VARIABLES - NUMBER_FIXED_EFFECTS));
      
      for (index_i = 0; index_i < NUMBER_INTERVALS; index_i++) {
        for (index_j = 0; index_j < NUMBER_FIXED_EFFECTS; index_j++) {
          MATRIX_SOLUTION_FIXED(index_i, index_j) = MATRIX_SOLUTION(index_i, index_j);
        }
        for (index_j = 0; index_j < (NUMBER_VARIABLES - NUMBER_FIXED_EFFECTS); index_j++) {
          MATRIX_SOLUTION_RANDOM(index_i, index_j) = MATRIX_SOLUTION(index_i, NUMBER_FIXED_EFFECTS + index_j);
        }
      }
      
      return List::create(Named("fixed_effects")  = MATRIX_SOLUTION_FIXED,
                          Named("random_effects") = MATRIX_SOLUTION_RANDOM,
                          Named("lambda")         = VECTOR_LAMBDA,
                          Named("iterations")     = VECTOR_ITERATIONS,
                          Named("rel_acc")        = EPSILON_CONVERGENCE,
                          Named("max_iter")       = ITERATION_MAX,
                          Named("gamma_bls")      = GAMMA,
                          Named("xi")             = PROPORTION_XI,
                          Named("loops_lambda")   = NUMBER_INTERVALS);
    }
  }
}
