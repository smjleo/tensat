#include "rust/cxx.h"
#include "tensat/src/input.rs.h"
#include <utility>
#include <bits/stdc++.h>

int main() {
    auto graphBox = new_converter();

    int dims[2] = {1024, 1024};
    auto input_slice = rust::Slice<const int32_t>{dims, 2};

    auto inp = graphBox->new_input(input_slice);
    auto relu = graphBox->relu(*inp);

    graphBox->print_rec_expr();
}
