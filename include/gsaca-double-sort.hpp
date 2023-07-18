#pragma once

#include "gsaca-double-sort/sequential/gsaca-ds.hpp"
#include "gsaca-double-sort/sequential/gsaca-ds-hash.hpp"

template<typename index_type, typename value_type>
static void gsaca_ds1(value_type const *const text, 
                      index_type *const sa,
                      size_t const n) {
  gsaca_lyndon::gsaca_ds1(text, sa, n);
}

template<typename index_type, typename value_type>
static void gsaca_ds2(value_type const *const text, 
                      index_type *const sa,
                      size_t const n) {
  gsaca_lyndon::gsaca_ds2(text, sa, n);
}

template<typename index_type, typename value_type>
static void gsaca_ds3(value_type const *const text, 
                      index_type *const sa,
                      size_t const n) {
  gsaca_lyndon::gsaca_ds3(text, sa, n);
}


template<typename index_type, typename value_type>
static void gsaca_dsh(value_type const *const text, 
                      index_type *const sa,
                      size_t const n) {
  gsaca_lyndon::gsaca_hash_ds(text, sa, n);
}