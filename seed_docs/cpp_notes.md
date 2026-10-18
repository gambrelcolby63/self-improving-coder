# C++17 quick notes (written for this repo, MIT)

* `std::nth_element(v.begin(), v.begin()+k, v.end())` puts the k-th smallest (0-indexed) at position k in O(n).
* `std::gcd` / `std::lcm` live in `<numeric>` (C++17). Use `long long` and `llabs` for negatives.
* `std::istringstream in(s); while (in >> w)` splits on whitespace and skips repeats.
* Bracket matching: push openers on a `std::vector<char>` stack, pop and compare on closers.
* Bit counting: `__builtin_popcount(x)` or DP `bits[i] = bits[i >> 1] + (i & 1)`.
* Compile with warnings: `g++ -std=c++17 -O2 -Wall -Wextra main.cpp -o main`.
* Prefer `const std::string&` parameters; return by value (move semantics make it cheap).
* `std::unordered_map<int,int>` gives O(1) average lookup for two-sum style problems.
