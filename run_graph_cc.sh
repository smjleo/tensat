g++ -std=c++20 -o tensatcpp src/bert.cc -I target/cxxbridge -I target/debug/build/tensat/out -L target/debug -ltensat -lc++ -lc++abi && ./tensatcpp
