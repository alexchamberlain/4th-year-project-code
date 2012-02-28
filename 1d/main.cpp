#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
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
      if(i != 1)
	m(i, i-1) = -ih2;
      m(i,i) = 2*ih2 + sigma;
      if(i != m.size1()-2)
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
    for(int i = 1; i < N; ++i) {
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
    for(int i = 1; i < N; ++i) {
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
    for(int i = 1; i < A.size1()-1; ++i) {
      w[i] *= omega;
      w[i] /= A(i,i);
    }
    u -= w;
  }
}

template<int nu, class M, class V>
void gauss_seidel(const M& A, V& u, const V& f);

template<>
void gauss_seidel(
  const ublas::banded_matrix<double, column_major, unbounded_array<double> >& A,
  vr& u, const vr& f) {
  v w(u.size());

  assert(A.size1() == A.size2());
  assert(u.size() == A.size1());
  assert(f.size() == A.size1());

  for(int i = 0; 
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

        // Clear the vector on the lower level.
	for(int i = 0; i < uh[l-1].size(); ++i) {
          uh[l-1][i] = 0.0;
	}
	
        // Restrict residual to grid below.
	p.restrict(rh[l], fh[l-1]);
      }

      // Invert A on coarsest grid.
      uh[0](1) = fh[0](1)/A[0](1,1);

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
};

class experiment {
  public:
    int maxl;
    std::vector<experiment_iteration> iterations;

    experiment(int _maxl) : maxl(_maxl) {
      iterations.reserve(maxl);
    }
};

std::ostream & operator << (std::ostream & out, const experiment & e) {
  out << "Multigrid solution on level " << e.maxl << "\n";
  out << "  i          ||r||_h       ||r||_inf          ||e||_h        ||e||_inf\n";
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
    out << std::endl;
  }

  return out;
}

void test_multigrid(problem & p, int level) {
  int N = (1 << (level + 1));
  multigrid<problem, 
            jacobi<2, problem::t_matrix, vr>,
	    jacobi<2, problem::t_matrix, vr> > g(level, p);
  v ur(N + 1);
  v r(N + 1);
  v e;

  experiment exp(level);
  std::vector<experiment_iteration> & eis = exp.iterations;

  ur.clear();
  p.u(ur, level);

  const double h = 1/(static_cast<double>(N));
  int i = 0;
  while(1) {
    g.solve();

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

    if(i > 0) {
      ei.l2rf  = ei.l2r/eis[i-1].l2r;
      ei.infrf = ei.infr/eis[i-1].infr;

      ei.l2ef  = ei.l2e/eis[i-1].l2e;
      ei.infef = ei.infe/eis[i-1].infe;
    }

    exp.iterations.push_back(ei);

    if(i > 0 && ei.l2rf > 0.5)
      break;

    ++i;
  }

  std::cout << exp << std::endl;
}

int main(int argc, char * argv[]) {
  const int maxl = 9;
  problem p(1, 0.0, 1.0);
  
  for(int i = 3; i <= maxl; ++i) {
    test_multigrid(p, i);
  }
}
