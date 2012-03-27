
class problem {
  public:
  typedef mapped_matrix<double> t_matrix;

  int k;
  int l;
  double C;

  problem(int _k, int _l, double _C) : k(_k), l(_l), C(_C)
    {/* Empty */}

  /*size_t global(size_t x, size_t y, size_t N) {
    return y * (N+1) + x;
  }*/

  void init_matrix(t_matrix & m, int l) {
    size_t N = (1 << (l+1));
    double ih  = static_cast<double>(N);
    double ih2 = ih * ih;
    size_t dim = (N + 1) * (N + 1); // (N+1)^d

    /*int ih2 = ih * ih;

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
    m(m.size1()-1, m.size1()-1) = 1;*/

    m.resize(dim, dim, false);
    m.clear();

    size_t idx = 0; // Global index

    /* The following relies on lexicographic ordering by lines of constant i */
    
    for(int i = 0; i <= N; ++i) {
      for(int j = 0; j <= N; ++j) {
        if(i == 0 || i == N || j == 0 || j == N) {
	  m(idx, idx) = 1.0;
	} else {
          m(idx, idx-1) = -ih2;
	  m(idx, idx)   = 2.0*ih2;
          m(idx, idx+1) = -ih2;
          m(idx, idx-N-1) = -ih2;
	  m(idx, idx+N+1) = -ih2;
	}

	++idx;
      }
    }
  }

  // l is r's level number...
  template<class V>
  void restrict(int l, V& r, V& rp) {
    rp.clear();

    unsigned int N  = (1 << (l+1));
    unsigned int Np = (1 << l);

    size_t idxp = 0;
    for(int i = 0; i <= Np; ++i) {
      size_t idx  = i * N;
      for(int j = 0; j <= Np; ++j) {
        if(i == 0 || i == Np || j == 0 || j == Np) {
	  // Do nothing...
	  //rp(idxp) = 0.0;
	} else {
	  rp(idxp) += r(idx-1-N-1); // 2i-1, 2j-1
	  rp(idxp) += r(idx-1+N+1); // 2i-1, 2j+1
	  rp(idxp) += r(idx+1-N-1); // 2i+1, 2j-1
	  rp(idxp) += r(idx+1+N+1); // 2i+1, 2j+1
	  
	  rp(idxp) /= 2.0;

	  rp(idxp) += r(idx  -N-1); // 2i, 2j-1
	  rp(idxp) += r(idx  +N+1); // 2i, 2j+1
	  rp(idxp) += r(idx-1);     // 2i-1, 2j
          rp(idxp) += r(idx+1);     // 2i+1, 2j

	  rp(idxp) /= 2.0;

	  rp(idxp) += r(idx);

	  rp(idxp) /= 4.0;
	}

	++idxp;
	idx += 2;
      }
    }
  }

  template<class V>
  void prolongate(int l, V& ep, V& e) {
    e.clear();

    unsigned int N  = (1 << (l+1));
    unsigned int Np = (1 << l);

    size_t idxp = 0;
    for(int i = 0; i <= Np; ++i) {
      size_t idx  = i * N;
      for(int j = 0; j <= Np; ++j) {
        if(i == 0 || i == Np || j == 0 || j == Np) {
	  // Do nothing...
	  //rp(idxp) = 0.0;
	} else {
	  e(idx) = ep(idxp);

	  e(idx+N+1) = (ep(idxp)+ep(idxp+Np+1))/2.0;
	  e(idx+1)   = (ep(idxp)+ep(idxp+1))/2.0;

	  e(idx+1+N+1) = (ep(idxp) + ep(idxp+1) + ep(idxp+Np+1)+ep(idxp+1+Np+1))/4.0;
	}

	++idxp;
	idx += 2;
      }
    }
  }
  
  double u(double x) {
    return (C*sin(k*M_PI*x)*sin(l*M_PI*y))/(M_PI*M_PI*(k*k + l*l));
  }

  void u(v & ur, int l) {
    int N = (1 << (l+1));
    double h = 1/(static_cast<double>(N));
    for(int i = 1; i < N; ++i) {
      ur[i] = u(i * h);
    }
  }

  double f(double x) {
    return C*sin(k*M_PI*x)*sin(l*M_PI*y);
  }

  void f(vr & fh, int l) {
    int N = (1 << (l+1));
    double h = 1/(static_cast<double>(N));
    for(int i = 1; i < N; ++i) {
      fh[i] = f(i * h);
    }
  }
};

