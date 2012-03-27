
template<class M, class V>
void jacobi(const M& A, V& u, const V& f) {
  const double omega = 2.0/3.0;
  v w(u);

  assert(A.size1() == A.size2());
  assert(u.size() == A.size1());
  assert(f.size() == A.size1());

  w = prod(A, u);
  w -= f;
  for(int i = 1; i < A.size1()-1; ++i) {
    w[i] *= omega;
    w[i] /= A(i,i);
  }
  u -= w;
}

template<class M, class V>
void gauss_seidel(const M& A, V& u, const V& f);

template<>
void gauss_seidel<ublas::banded_matrix<double, column_major, unbounded_array<double> >, vr> (
  const ublas::banded_matrix<double, column_major, unbounded_array<double> >& A,
  vr& u, const vr& f) {
  v w(u.size());

  assert(A.size1() == A.size2());
  assert(u.size() == A.size1());
  assert(f.size() == A.size1());
  
  w = prod(A,u);
  w -= f;

  for(int i = 0; i < u.size(); ++i) {
    w[i] = 0;
    for(int j = i+1; (j < f.size()) && (j <= i + A.upper()); ++j)
      w[i] -= A(i,j)*u[j];
  }

  w += f;

  for(int i = 0; i < u.size(); ++i) {
    u[i] = w[i];
    for(int j = std::max(static_cast<size_t>(0), i-A.lower()); j < i; ++j) {
      u[i] -= A(i,j)*u[j];
    }
    u[i] /= A(i,i);
  }
}

