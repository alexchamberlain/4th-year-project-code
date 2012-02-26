#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

using boost::numeric::ublas::range;
using boost::numeric::ublas::banded_matrix;
using boost::numeric::ublas::column_major;
using boost::numeric::ublas::unbounded_array;
using boost::numeric::ublas::matrix_range;
typedef boost::numeric::ublas::vector<double> v;
typedef boost::numeric::ublas::vector_range<boost::numeric::ublas::vector<double> > vr;

class problem {
  public:
  typedef banded_matrix<double, column_major, unbounded_array<double> > t_matrix;
  typedef matrix_range<t_matrix> t_matrix_range;
  size_t matrix_dim;

  int k;
  double sigma;
  double C;

  problem(int _k, double _sigma, double _C) : k(_k), sigma(_sigma), C(_C) {/* Empty */}
  void init_matrix(t_matrix & m, t_matrix_range ** mr, int maxl) {
    size_t N = (1 << (maxl + 1)) + 1;
    matrix_dim = N;

    m.resize(N, N, 1, 1);
    for(int i = 0; i < m.size1(); ++i) {
      if(i != 0)
        m(i, i-1) = 1;
      m(i,i) = 2;
      if(i+1 != m.size1())
        m(i, i+1) = 1;
    }

    *mr = static_cast<t_matrix_range *>(malloc(sizeof(t_matrix_range)*(maxl+1)));
    for(size_t i = 0; i <= maxl; ++i) {
      int stop = (1 << (i+1)) + 1;
      range r(0, stop);
      ::new(*mr + i) t_matrix_range(m, r, r);
    }
  }
  
  double u(double x) {
    return (C*sin(k*M_PI*x))/(M_PI*M_PI*k*k + sigma);
  }

  double du(double x) {
    return (C*k*M_PI*cos(k*M_PI*x))/(M_PI*M_PI*k*k + sigma);
  }

  double d2u(double x) {
    return (-C*k*k*M_PI*M_PI*sin(k*M_PI*x))/(M_PI*M_PI*k*k + sigma);
  }

  double f(double x) {
    return C*sin(k*M_PI*x);
  }

  void f(vr & fh, int l) {
    int N = (1 << (l+1)) + 1;
    double h = 1/(static_cast<double>(N-1));
    for(int i = 0; i < N; ++i) {
      fh[i] = f(i * h);
    }
  }
};

template<class prob>
class multigrid {
  public:
    prob & p;
    int maxl;
    size_t N;
    v raw_uh;
    v raw_fh;

    vr *uh;
    vr *fh;

    typename prob::t_matrix raw_A;
    typename prob::t_matrix_range *A;

    multigrid(int _maxl, prob & _p) : maxl(_maxl), p(_p) {
      N = (1 << (maxl + 2)) + maxl - 1;

      raw_uh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));
      raw_fh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));

      raw_uh.clear();
      raw_fh.clear();

      std::cout << raw_uh << std::endl;
      std::cout << raw_fh << std::endl;

      uh = static_cast<vr *>(malloc(sizeof(vr)*(maxl+1)));
      fh = static_cast<vr *>(malloc(sizeof(vr)*(maxl+1)));

      for(size_t i = 0; i <= maxl; ++i) {
        unsigned int start = (1 << (i + 1)) + i - 2;
	unsigned int stop  = (1 << (i + 2)) + i - 1;
        range r(start, stop);
	::new(uh + i) vr(raw_uh, range(start, stop));
	::new(fh + i) vr(raw_fh, range(start,stop));
      }

      p.f(*(fh + maxl), maxl);

      p.init_matrix(raw_A, &A, maxl);
    }

    ~multigrid() {
      free(uh);
      free(fh);
    }
};

int main(int argc, char * argv[]) {
  problem p(1, 1.0, 1.0);
  multigrid<problem> g(3, p);

  std::cout << "Max Level:             " << g.maxl << "\n"
            << "Allocated Vector Size: " << g.N << "\n"
	    << "Level Vectors:\n";

  for(int i = 0; i <= g.maxl; ++i) {
    std::cout << "(" << i << ", " << g.uh[i] << ", " << g.fh[i] << ")\n";
  }

  std::cout << "Allocated Matrix Size: " << p.matrix_dim << "*" << p.matrix_dim << "\n";
  std::cout << "Level Matrices:\n";
  for(int i = 0; i <= g.maxl; ++i) {
    std::cout << "(" << i << ", " << g.A[i] << ")\n";
  }
}
