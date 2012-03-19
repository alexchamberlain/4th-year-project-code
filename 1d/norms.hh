
template<class V>
double norm_h(double h, V& u) {
  return sqrt(h*inner_prod(u,u));
}

