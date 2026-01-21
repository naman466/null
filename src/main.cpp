#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::vector<std::string> args(argv + 1, argv + argc);

  if (args.empty()) {
    std::cerr << "Usage: nullc [model_path]\n";
    return 1;
  }

  std::cout << "[Null] Initializing Compiler for: " << args[0] << "\n";
  std::cout << "[Null] Target Architecture: Generic C\n";

  return 0;
}
