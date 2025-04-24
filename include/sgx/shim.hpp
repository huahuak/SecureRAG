#ifndef SHIM_HPP
#define SHIM_HPP

#include <sgx_trts.h>

namespace std {

int rand() {
  int buff = 0;
  sgx_read_rand((unsigned char *)&buff, sizeof(buff));
  return buff;
}

} // namespace std

#endif
