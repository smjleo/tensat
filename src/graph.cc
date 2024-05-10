#include "rust/cxx.h"
#include "tensat/src/input.rs.h"
#include <utility>
#include <bits/stdc++.h>


int main() {
    auto graphBox = new_converter();
    
    int dims[2] = {1024, 1024};
    auto input_slice = rust::Slice<const int32_t>{dims, 2};
    
    auto inp = graphBox->new_input(input_slice);
    auto relu = graphBox->relu(std::move(inp));

    graphBox->print_rec_expr();
}


/*
class CppGraphConverterWrapper
{
public:
    CppGraphConverterWrapper()
    {
        converter = std::make_unique<::rust::Box<::CppGraphConverter>>();
    }

    ::rust::Box<::TensorInfo> newInput(const std::vector<int32_t> &dims)
    {
        ::rust::Slice<const int32_t> rust_dims(dims.data(), dims.size());
        return (*converter)->new_input(rust_dims);
    }

    ::rust::Box<::TensorInfo> relu(::rust::Box<::TensorInfo> input)
    {
        return (*converter)->relu(std::move(input));
    }

private:
    std::unique_ptr<::rust::Box<::CppGraphConverter>> converter;
};

void test()
{
    CppGraphConverterWrapper graphConverter;

    std::vector<int32_t> dims = {1, 3, 224, 224};
    auto tensor = graphConverter.newInput(dims);
    auto result = graphConverter.relu(std::move(tensor));
}

int main()
{
    test();
    return 0;
}
*/