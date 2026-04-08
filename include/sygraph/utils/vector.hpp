/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>
#include <sygraph/utils/memory.hpp>


namespace sygraph {

/**
 * @todo Remove it, it might be not necessary.
 */
template<typename T>
class Vector {
public:
  Vector(sycl::queue& q, size_t size) : _q(q), _data(sycl::malloc_shared<T>(size, q)), _size(size) {}

  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  Vector(Vector&& other) noexcept : _q(other._q), _data(other._data), _size(other._size) {
    other._data = nullptr;
    other._size = 0;
  }

  Vector& operator=(Vector&&) = delete;

  ~Vector() { memory::detail::releaseUSM(_data, _q); }

  T* getData() const { return _data; }

  size_t size() const { return _size; }

private:
  sycl::queue& _q;
  T* _data;
  size_t _size;
};

} // namespace sygraph
