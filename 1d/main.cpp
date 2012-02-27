#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

namespace ublas = boost::numeric::ublas;

using ublas::range;
using ublas::banded_matrix;
using ublas::column_major;
using ublas::unbounded_array;
using ublas::matrix_range;
typedef ublas::vector<double> v;
typedef ublas::vector_range<ublas::vector<double> > vr;
typedef ublas::permutation_matrix<std::size_t> pmatrix;

class problem {
  public:
  typedef banded_matrix<double, column_major, unbounded_array<double> > t_matrix;

  int k;
  double sigma;
  double C;

  problem(int _k, double _sigma, double _C) : k(_k), sigma(_sigma), C(_C)
    {/* Empty */}

  void init_matrix(t_matrix & m, pmatrix & pm, int l) {
    int ih = (1 << (l+1));
    size_t N = (1 << (l+1)) + 1;

    int ih2 = ih * ih;

    m.resize(N, N, 1, 1);
    m(0,0) = 1;
    for(int i = 1; i < m.size1()-1; ++i) {
      m(i, i-1) = -ih2;
      m(i,i) = 2*ih2 + sigma;
      m(i, i+1) = -ih2;
    }
    m(m.size1()-1, m.size1()-1) = 1;

    if(l == 0) {
      ublas::lu_factorize(m, pm);
    }
  }

  void restrict(vr& r, vr& rp) {
    for(int i = 1; i < rp.size()-1; ++i) {
      rp[i] = (r[2*i-1] + 2*r[2*i] + r[2*i+1])/4.0;
    }
  }

  void prolongate(vr& ep, vr& e) {
    e[1] = ep[1]/2.0;

    for(int i = 2; i < e.size()-1; i += 2) {
      e[i] = ep[i/2];
    }

    for(int i = 3; i < e.size()-1; i += 2) {
      e[i] = (ep[(i-1)/2] + ep[(i+1)/2])/2.0;
    }
  }
  
  double u(double x) {
    return (C*sin(k*M_PI*x))/(M_PI*M_PI*k*k + sigma);
  }

  void u(v & ur, int l) {
    int N = (1 << (l+1)) + 1;
    double h = 1/(static_cast<double>(N-1));
    for(int i = 0; i < N; ++i) {
      ur[i] = u(i * h);
    }
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

template<int nu, class M, class V>
void jacobi(const M& A, V& u, const V& f) {
  const double omega = 2.0/3.0;
  v w(u);

  std::cout << "A: " << A << std::endl;
  std::cout << "u: " << u << std::endl;

  for(int i = 0; i < nu; ++i) {
    w = prod(A, u);
    w -= f;
    for(int i = 0; i < A.size1(); ++i) {
      w[i] *= omega/A(i,i);
    }
    u -= w;
    std::cout << "u: " << u << std::endl;
  }
}

template<class prob,
	 void presmooth(const typename prob::t_matrix& M, vr& u, const vr& f),
	 void postsmooth(const typename prob::t_matrix& M, vr& u, const vr& f)>
class multigrid {
  public:
    prob & p;
    int maxl;
    size_t N;
    v raw_uh;
    v raw_rh;
    v raw_fh;

    vr *uh;
    vr *rh;
    vr *fh;

    typedef typename prob::t_matrix t_matrix;

    t_matrix *A;
    pmatrix pm;

    multigrid(int _maxl, prob & _p) : maxl(_maxl), p(_p), pm(3) {
      N = (1 << (maxl + 2)) + maxl - 1;

      raw_uh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));
      raw_rh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));
      raw_fh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));

      raw_uh.clear();
      raw_rh.clear();
      raw_fh.clear();

      uh = static_cast<vr *>(malloc(sizeof(vr)*(maxl+1)));
      rh = static_cast<vr *>(malloc(sizeof(vr)*(maxl+1)));
      fh = static_cast<vr *>(malloc(sizeof(vr)*(maxl+1)));
      A  = static_cast<t_matrix*>(malloc(sizeof(t_matrix)*(maxl+1)));

      for(size_t i = 0; i <= maxl; ++i) {
        unsigned int start = (1 << (i + 1)) + i - 2;
	unsigned int stop  = (1 << (i + 2)) + i - 1;
        range r(start, stop);
	::new(uh + i) vr(raw_uh, r);
	::new(rh + i) vr(raw_rh, r);
	::new(fh + i) vr(raw_fh, r);

	::new(A + i) t_matrix();
	p.init_matrix(A[i], pm, i);
      }

      p.f(*(fh + maxl), maxl);
    }

    ~multigrid() {
      for(size_t i = 0; i <= maxl; ++i) {
	(uh + i)->~vr();
	(rh + i)->~vr();
	(fh + i)->~vr();

	(A + i)->~t_matrix();
      }

      free(uh);
      free(rh);
      free(fh);
      free(A);
    }

    void solve() {
      for(int l = maxl; l > 0; --l) {
        // Smooth error
        presmooth(A[l], uh[l], fh[l]);
	
	// Compute residual
	rh[l] = prod(A[l], uh[l]);
	rh[l] *= -1;
        rh[l] += fh[l];

	p.restrict(rh[l], fh[l-1]);
      }

      std::cout << fh[0] << std::endl;
      std::cout << A[0] << std::endl;
      std::cout << pm << std::endl;

      uh[0] = fh[0];
      lu_substitute(A[0], pm, uh[0]);

      std::cout << uh[0] << std::endl;



      for(int l = 1; l <= maxl; ++l) {
        p.prolongate(uh[l-1], rh[l]);

	uh[l] += rh[l];

        postsmooth(A[l], uh[l], fh[l]);
      }
    }
};

int main(int argc, char * argv[]) {
  problem p(1, 0.0, 1.0);
  multigrid<problem, 
            jacobi<1, problem::t_matrix, vr>,
	    jacobi<1, problem::t_matrix, vr> > g(3, p);

  v ur((1 << 4) + 1);
  p.u(ur, 3);

  std::cout << "Max Level:             " << g.maxl << "\n"
            << "Allocated Vector Size: " << g.N << "\n"
	    << "Level Vectors:\n";

  for(int i = 0; i <= g.maxl; ++i) {
    std::cout << "(" << i << ", " << g.uh[i] << ", " << g.fh[i] << ")\n";
  }

  std::cout << "Level Matrices:\n";
  for(int i = 0; i <= g.maxl; ++i) {
    std::cout << "(" << i << ", " << g.A[i] << ")\n";
  }

  /*std::cout << "Jacobi test:" << "\n";
  std::cout << g.uh[3] << "\n";
  jacobi<2, problem::t_matrix, vr>(g.A[3], g.uh[3], g.fh[3]);
  std::cout << g.uh[3] << "\n";*/
  for(int i = 0; i < 20; ++i) {
    g.solve();
    std::cout << g.uh[3] << "\n";
  }

  std::cout << "Calculated u: " << ur << std::endl;
}
