#include <iostream>
#include <string>
#include <boost/type_index.hpp>


template<typename T>
void f(T param) {
  using boost::typeindex::type_id_with_cvr;
  std::cout << "T     = " << type_id_with_cvr<T>().pretty_name() << std::endl;
  std::cout << "param = " << type_id_with_cvr<decltype(param)>().pretty_name() << std::endl;
}

template<typename T>
void g(T& param) {
  using boost::typeindex::type_id_with_cvr;
  std::cout << "T     = " << type_id_with_cvr<T>().pretty_name() << std::endl;
  std::cout << "param = " << type_id_with_cvr<decltype(param)>().pretty_name() << std::endl;
}

template<typename T>
void h(T* param) {
  using boost::typeindex::type_id_with_cvr;
  std::cout << "T     = " << type_id_with_cvr<T>().pretty_name() << std::endl;
  std::cout << "param = " << type_id_with_cvr<decltype(param)>().pretty_name() << std::endl;
}


template<typename T>
void k(T&& param) {
  using boost::typeindex::type_id_with_cvr;
  std::cout << "T     = " << type_id_with_cvr<T>().pretty_name() << std::endl;
  std::cout << "param = " << type_id_with_cvr<decltype(param)>().pretty_name() << std::endl;
}


void g() {}

int main() {
  // not a reference or pointer
  int a = 1;
  const int ca = 2;

  // Reference 
  int& ra = a;
  const int& rca = 2;
  int&& rva = 2; // Rvalue reference, it's lvalue itself
  
  // Pointer 
  int* p1 = &a;
  const int* p2 = p1;
  int* const p3 = &a;
  const int* const p4 = p1;

  // array 
  int arr[10] = {1};

}
