#include "sentiment/text/tokenizer.hpp"
#include <bits/stdc++.h>

using namespace std;

int main() {

    sentiment::Tokenizer tokenizer;

    const auto tokens =
        tokenizer.tokenize(
            "This application is absolutely great!"
        );

    cout << "Tokens:\n";

    for (const auto& token : tokens) {
        cout << token << '\n';
    }

    return 0;
}