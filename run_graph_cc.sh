g++ -std=c++11 -o tensatcpp src/graph.cc -I target/cxxbridge -Ltarget/debug -ltensat -ltaso_runtime && ./tensatcpp
