#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <sys/resource.h>

namespace ublas = boost::numeric::ublas;

using ublas::range;
using ublas::banded_matrix;
using ublas::column_major;
using ublas::unbounded_array;
using ublas::matrix_range;
typedef ublas::vector<double> v;
typedef ublas::vector_range<ublas::vector<double> > vr;
typedef ublas::permutation_matrix<std::size_t> pmatrix;

#include "problem_1d.hh"
#include "norm.hh"
#include "smoothing_operators.hh"
#include "multigrid.hh"

class experiment_iteration {
  public:
    int i;
    double l2r;
    double l2rf;
    double infr;
    double infrf;
    double l2e;
    double l2ef;
    double infe;
    double infef;
    int time;
};

class experiment {
  public:
    int maxl;
    struct rusage start;
    int time;
    std::vector<experiment_iteration> iterations;

    experiment(int _maxl) : maxl(_maxl) {
      iterations.reserve(maxl);
    }
};

std::ostream & operator << (std::ostream & out, const experiment & e) {
  out << "Multigrid solution on level " << e.maxl << "\n";
  out << "  i          ||r||_h       ||r||_inf          ||e||_h        ||e||_inf Accrued Time\n";
  for(std::vector<experiment_iteration>::const_iterator i = e.iterations.begin();
      i != e.iterations.end(); ++i) {
    out << " "  << std::setw(2) << i->i;
    out << " "  << std::right << std::scientific << std::setw(7) << std::setprecision(3) << i->l2r
        << " (" << std::right << std::fixed      << std::setw(4) << std::setprecision(2) << i->l2rf << ")";
    out << " "  << std::right << std::scientific << std::setw(7) << std::setprecision(3) << i->infr
        << " (" << std::right << std::fixed      << std::setw(4) << std::setprecision(2) << i->infrf << ")";
    out << " "  << std::right << std::scientific << std::setw(7) << std::setprecision(3) << i->l2e
        << " (" << std::right << std::fixed      << std::setw(4) << std::setprecision(2) << i->l2ef << ")";
    out << " "  << std::right << std::scientific << std::setw(7) << std::setprecision(3) << i->infe
        << " (" << std::right << std::fixed      << std::setw(4) << std::setprecision(2) << i->infef << ")";
    out << " "  << std::right << std::fixed      << std::setw(5) << i->time;
    out << std::endl;
  }

  return out;
}

template<int nu_1, int nu_2, 
         void smooth(const typename problem::t_matrix& M, vr& u, const vr & f)>
int test_multigrid(problem & p, int level) {
  int N = (1 << (level + 1));
  multigrid<problem, nu_1, smooth, nu_2, smooth> g(level, p);
  v ur(N + 1);
  v r(N + 1);
  v e;

  struct rusage end;

  experiment exp(level);
  std::vector<experiment_iteration> & eis = exp.iterations;

  ur.clear();
  p.u(ur, level);

  const double h = 1/(static_cast<double>(N));
  int i = 0;
  getrusage(RUSAGE_SELF, &exp.start);
  while(1) {
    g.solve();

    getrusage(RUSAGE_SELF, &end);

    // Calculate the residual at the finest grid.
    r = prod(g.A[level], g.uh[level]);
    r *= -1;
    r += g.fh[level];

    // Calculate the error on the finest grid.
    e = ur - g.uh[level];
    
    // Record statistics.
    experiment_iteration ei;

    ei.i = i+1;

    ei.l2r  = norm_h<v>(h,r);
    ei.infr = norm_inf(r);

    ei.l2e  = norm_h<v>(h,e);
    ei.infe = norm_inf(e);

    if(i == 0) {
      ei.l2rf  = 0.0;
      ei.infrf = 0.0;

      ei.l2ef  = 0.0;
      ei.infef = 0.0;
    } else {
      ei.l2rf  = ei.l2r/eis[i-1].l2r;
      ei.infrf = ei.infr/eis[i-1].infr;

      ei.l2ef  = ei.l2e/eis[i-1].l2e;
      ei.infef = ei.infe/eis[i-1].infe;
    }

    ei.time = (end.ru_utime.tv_sec*1e6 + end.ru_utime.tv_usec) - (exp.start.ru_utime.tv_sec*1e6 + exp.start.ru_utime.tv_usec);
    exp.time = ei.time;

    exp.iterations.push_back(ei);

    if(i > 0 && ei.l2rf > 0.5)
      break;

    ++i;
  }

  std::cout << exp << std::endl;
  return exp.time;
}

template<int nu_1, int nu_2, 
         void smooth(const typename problem::t_matrix& M, vr& u, const vr & f)>
int test_full_multigrid(problem & p, int level) {
  int N = (1 << (level + 1));
  multigrid<problem, nu_1, smooth, nu_2, smooth> g(level, p);
  v ur(N + 1);
  v r(N + 1);
  v e;

  struct rusage end;

  experiment exp(level);
  std::vector<experiment_iteration> & eis = exp.iterations;

  ur.clear();
  p.u(ur, level);

  const double h = 1/(static_cast<double>(N));
  int i = 0;
  getrusage(RUSAGE_SELF, &exp.start);
  while(1) {
    if(i == 0) {
      g.full_solve();
    } else {
      g.solve();
    }

    getrusage(RUSAGE_SELF, &end);

    // Calculate the residual at the finest grid.
    r = prod(g.A[level], g.uh[level]);
    r *= -1;
    r += g.fh[level];

    // Calculate the error on the finest grid.
    e = ur - g.uh[level];
    
    // Record statistics.
    experiment_iteration ei;

    ei.i = i+1;

    ei.l2r  = norm_h<v>(h,r);
    ei.infr = norm_inf(r);

    ei.l2e  = norm_h<v>(h,e);
    ei.infe = norm_inf(e);

    if(i == 0) {
      ei.l2rf  = 0.0;
      ei.infrf = 0.0;

      ei.l2ef  = 0.0;
      ei.infef = 0.0;
    } else {
      ei.l2rf  = ei.l2r/eis[i-1].l2r;
      ei.infrf = ei.infr/eis[i-1].infr;

      ei.l2ef  = ei.l2e/eis[i-1].l2e;
      ei.infef = ei.infe/eis[i-1].infe;
    }

    ei.time = (end.ru_utime.tv_sec*1e6 + end.ru_utime.tv_usec) - (exp.start.ru_utime.tv_sec*1e6 + exp.start.ru_utime.tv_usec);
    exp.time = ei.time;

    exp.iterations.push_back(ei);

    if(i > 0 && ei.l2rf > 0.5)
      break;

    ++i;
  }

  std::cout << exp << std::endl;
  return exp.time;
}

int main(int argc, char * argv[]) {
  using std::pair;

  const int maxl = 13;
  problem p(1, 0.0, 1.0);
  std::map<int, int> jacobi_time;
  std::map<int, int> full_jacobi_time;
  std::map<int, int> gauss_seidel_time;
  std::map<int, int> full_gauss_seidel_time;
  
  for(int i = 3; i <= maxl; ++i) {
    jacobi_time.insert(pair<int, int>(i, test_multigrid<2, 2, jacobi<problem::t_matrix, vr> >(p, i)));
    full_jacobi_time.insert(pair<int, int>(i, test_full_multigrid<2, 2, jacobi<problem::t_matrix, vr> >(p, i)));
    gauss_seidel_time.insert(pair<int, int>(i, test_multigrid<2, 2, gauss_seidel<problem::t_matrix, vr> >(p, i)));
    full_gauss_seidel_time.insert(pair<int, int>(i, test_full_multigrid<2, 2, gauss_seidel<problem::t_matrix, vr> >(p, i)));
  }

  /*for(int i = 4; i <= maxl; ++i) {
    std::cout << full_jacobi_time[i]/((double) full_jacobi_time[i-1]) << std::endl;
    std::cout << full_gauss_seidel_time[i]/((double) full_jacobi_time[i-1]) << std::endl;
  }*/
}
