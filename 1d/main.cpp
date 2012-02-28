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
    m.clear();
    m(0,0) = 1;
    for(int i = 1; i < m.size1()-1; ++i) {
      m(i, i-1) = -ih2;
      m(i,i) = 2*ih2 + sigma;
      m(i, i+1) = -ih2;
    }
    m(m.size1()-1, m.size1()-1) = 1;
  }

  template<class V>
  void restrict(V& r, V& rp) {
    for(int i = 1; i < rp.size()-1; ++i) {
      rp[i] = (r[2*i-1] + 2*r[2*i] + r[2*i+1])/4.0;
    }
  }

  template<class V>
  void prolongate(V& ep, V& e) {
    e[1] = ep[1]/2.0;

    for(int i = 2; i < e.size()-2; i += 2) {
      e[i] = ep[i/2];
    }

    for(int i = 3; i < e.size()-2; i += 2) {
      e[i] = (ep[(i-1)/2] + ep[(i+1)/2])/2.0;
    }

    e[e.size()-2] = ep[ep.size()-2]/2.0;
  }
  
  double u(double x) {
    return (C*sin(k*M_PI*x))/(M_PI*M_PI*k*k + sigma);
  }

  void u(v & ur, int l) {
    int N = (1 << (l+1));
    double h = 1/(static_cast<double>(N));
    for(int i = 1; i <= N; ++i) {
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
    int N = (1 << (l+1));
    double h = 1/(static_cast<double>(N));
    for(int i = 1; i <= N; ++i) {
      fh[i] = f(i * h);
    }
  }
};

template<int nu, class M, class V>
void jacobi(const M& A, V& u, const V& f) {
  const double omega = 2.0/3.0;
  v w(u);

  assert(A.size1() == A.size2());
  assert(u.size() == A.size1());
  assert(f.size() == A.size1());

  for(int i = 0; i < nu; ++i) {
    w = prod(A, u);
    w -= f;
    for(int i = 0; i < A.size1(); ++i) {
      w[i] *= omega/A(i,i);
    }
    u -= w;
  }
}

template<class V>
double norm_h(double h, V& u) {
  return sqrt(h*inner_prod(u,u));
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

      p.f(fh[maxl], maxl);
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
	
	//std::cout << fh[l] << std::endl;

        // Restrict residual to grid below.
	p.restrict(rh[l], fh[l-1]);
      }

      // Invert A on coarsest grid.
      if(maxl >= 8) {
        std::cout << fh[0] << std::endl;
      }
      uh[0](1) = fh[0](1)/A[0](1,1);
      if(maxl >= 8) {
        std::cout << uh[0] << std::endl;
      }

      for(int l = 1; l <= maxl; ++l) {
        // Prolongate error to grid above.
        p.prolongate(uh[l-1], rh[l]);
        //std::cout << rh[l] << std::endl;

        // Coarse grid correction.
	uh[l] += rh[l];

        // Smooth again.
        postsmooth(A[l], uh[l], fh[l]);
      }
    }
};

void test_multigrid(problem & p, int level) {
  int N = (1 << (level + 1));
  multigrid<problem, 
            jacobi<2, problem::t_matrix, vr>,
	    jacobi<2, problem::t_matrix, vr> > g(level, p);
  v ur(N + 1);
  v e;

  p.u(ur, level);

  std::cout << "Level Matrices:\n";
  for(int i = 0; i < level; ++i) {
    std::cout << g.A[i] << std::endl;
  }

  std::cout << "Multigrid solution on level " << level << ":\n";
  const double h = 1/(static_cast<double>(N));
  for(int i = 0; i < level; ++i) {
    g.solve();
    e = ur - g.uh[level];
    std::cout << "||e||_h = " << norm_h<v>(h, e) << std::endl;
    std::cout << "||e||_{\\infty} = " << norm_inf(e) << std::endl;
  }

  //std::cout << g.uh[level] << std::endl;
}

void test_restrict(problem & p) {
  v w(5);
  v wp(9);

  w[0] = 0;
  w[1] = 1;
  w[2] = 1;
  w[3] = 1;
  w[4] = 0;

  p.prolongate<v>(w, wp);
  p.restrict<v>(wp, w);

  std::cout << w << std::endl;

  wp[0] = 0;
  wp[1] = 1;
  wp[2] = 1;
  wp[3] = 1;
  wp[4] = 1;
  wp[5] = 1;
  wp[6] = 1;
  wp[7] = 1;
  wp[8] = 0;

  p.restrict<v>(wp, w);
  p.prolongate<v>(w, wp);

  std::cout << wp << std::endl;
}

int main(int argc, char * argv[]) {
  const int maxl = 10;
  problem p(1, 0.0, 1.0);
  
  //test_restrict(p);

  for(int i = 3; i <= maxl; ++i) {
    test_multigrid(p, i);
  }
}
