
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

