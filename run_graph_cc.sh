g++ -std=c++20 -o tensatcpp src/bert.cc -I target/cxxbridge -Ltarget/debug -ltensat -ltaso_runtime && ./tensatcpp
