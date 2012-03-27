
template<class prob,
         int nu_1,
	 void presmooth(const typename prob::t_matrix& M, vr& u, const vr& f),
	 int nu_2,
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
	for(int i = 0; i < nu_1; ++i)
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
	p.restrict(l, rh[l], fh[l-1]);
      }

      // Invert A on coarsest grid.
      uh[0](1) = fh[0](1)/A[0](1,1);

      for(int l = 1; l <= maxl; ++l) {
        // Prolongate error to grid above.
        p.prolongate(l-1, uh[l-1], rh[l]);
        //std::cout << rh[l] << std::endl;

        // Coarse grid correction.
	uh[l] += rh[l];

        // Smooth again.
	for(int i = 0; i < nu_2; ++i)
	  postsmooth(A[l], uh[l], fh[l]);
      }
    }

    void full_solve() {
      for(int tl = 0; tl <= maxl; ++tl) {
	p.f(fh[tl], tl);
	for(int l = tl; l > 0; --l) {
	  // Smooth error
	  for(int i = 0; i < nu_1; ++i)
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
	  p.restrict(l, rh[l], fh[l-1]);
	}

	// Invert A on coarsest grid.
	uh[0](1) = fh[0](1)/A[0](1,1);

	for(int l = 1; l <= tl; ++l) {
	  // Prolongate error to grid above.
	  p.prolongate(l-1, uh[l-1], rh[l]);
	  //std::cout << rh[l] << std::endl;

	  // Coarse grid correction.
	  uh[l] += rh[l];

	  // Smooth again.
	  for(int i = 0; i < nu_2; ++i)
	    postsmooth(A[l], uh[l], fh[l]);
	}

        if(tl < maxl) {
	  // Prolongate soln to grid above.
	  p.prolongate(tl, uh[tl], uh[tl+1]);
	}
      }
    }
};

