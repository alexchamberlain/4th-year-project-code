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

namespace ublas = boost::numeric::ublas;

using ublas::range;
using ublas::banded_matrix;
using ublas::column_major;
using ublas::unbounded_array;
using ublas::matrix_range;
typedef ublas::vector<double> v;
typedef ublas::vector_range<ublas::vector<double> > vr;
typedef permutation_matrix<std::size_t> pmatrix;

class problem {
  public:
  typedef banded_matrix<double, column_major, unbounded_array<double> > t_matrix;

  int k;
  double sigma;
  double C;

  problem(int _k, double _sigma, double _C) : k(_k), sigma(_sigma), C(_C) {/* Empty */}
  void init_matrix(t_matrix & m, pmatrix & pm, int l) {
    int ih = (1 << (l+1));
    size_t N = (1 << (l+1)) + 1;

    int ih2 = ih * ih;

    m.resize(N, N, 1, 1);
    for(int i = 0; i < m.size1(); ++i) {
      if(i != 0)
        m(i, i-1) = -ih2;
      m(i,i) = 2*ih2 + sigma;
      if(i+1 != m.size1())
        m(i, i+1) = -ih2;
    }

    if(l == 0) {
      lu_factorize(m, pm);
    }
  }

  void restrict(v& r, v& rp) {
    for(int i = 0; i < rp.size(); ++i) {
      rp[i] = (r[2*i] + 2*r[2*i+1] + r[2*i+2])/4.0;
    }
  }

  void prolongate(v& ep, v& e) {
    e[0] = ep[0]/2.0;

    for(int i = 1; i < e.size()-1; i += 2) {
      e[i] = ep[(i-1)/2];
    }

    for(int i = 2; i < e.size()-1; i += 2) {
      e[i] = (ep[(i-2)/2] + ep[i/2])/2.0;
    }

    e[e.size()-1] = ep[ep.size()-1]/2.0;
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

template<int nu, class M, class V>
void jacobi(const M& A, V& u, const V& f) {
  const double omega = 2.0/3.0;
  v w(u);

  std::cout << "u: " << u << std::endl;

  for(int i = 0; i < nu; ++i) {
    w = prod(A, u);
    w += f;
    for(int i = 0; i < A.size1(); ++i) {
      w[i] *= omega/A(i,i);
    }
    u -= w;
    std::cout << "u: " << u << std::endl;
  }
}

template<class prob>
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

    multigrid(int _maxl, prob & _p) : maxl(_maxl), p(_p) {
      N = (1 << (maxl + 2)) + maxl - 1;

      raw_uh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));
      raw_rh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));
      raw_fh.resize(N);// = static_cast<double *>(malloc(sizeof(double)*N));
      pm.resize(3);

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
	p.init_matrix(A[i], maxl);
      }

      p.f(*(fh + maxl), maxl);
    }

    ~multigrid() {
      for(size_t i = 0; i <= maxl; ++i) {
	(uh + i)->~vr();
	(fh + i)->~vr();

	(A + i)->~t_matrix();
      }

      free(uh);
      free(fh);
      free(A);
    }

    template<void presmooth(const t_matrix& M, v& u, const v& f),
             void postsmooth(const t_matrix& M, v& u, const v& f)>
    void solve() {
      int l = maxl;
      for(int l = maxl; l > 0; --l) {
        // Smooth error
        presmooth(A[l], uh[l], fh[l]);
	
	// Compute residual
	rh[l] = prod(A[l], uh[l]);
	rh[l] *= -1;
        rh[l] += fh[l];

	p.restrict(rh[l], rh[l-1]);
      }

      uh[0] = fh[0];
      lu_substitute(m, pm, uh[0]);

      for(l = 1; l <= maxl; ++l) {

        postsmooth(A[l], uh[l], fh[l]);
      }
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

  std::cout << "Level Matrices:\n";
  for(int i = 0; i <= g.maxl; ++i) {
    std::cout << "(" << i << ", " << g.A[i] << ")\n";
  }

  std::cout << "Jacobi test:" << "\n";
  std::cout << g.uh[3] << "\n";
  jacobi<2, problem::t_matrix, vr>(g.A[3], g.uh[3], g.fh[3]);
  std::cout << g.uh[3] << "\n";
}
