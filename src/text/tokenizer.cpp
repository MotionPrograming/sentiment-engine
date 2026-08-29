#include "sentiment/text/tokenizer.hpp"
using namespace std;

namespace sentiment {

bool Tokenizer::is_ascii_alpha(char c) noexcept {

    return
        (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z');
}

char Tokenizer::to_lower(char c) noexcept {

    if (c >= 'A' && c <= 'Z') {
        return static_cast<char>(c + ('a' - 'A'));
    }

    return c;
}

std::vector<std::string>
Tokenizer::tokenize(std::string_view text) const {

    std::vector<std::string> tokens;

    std::string current;

    current.reserve(32);

    for (char c : text) {

        if (is_ascii_alpha(c)) {

            current.push_back(to_lower(c));

        } else {

            if (!current.empty()) {

                tokens.emplace_back(std::move(current));

                current.clear();

                current.reserve(32);
            }
        }
    }

    if (!current.empty()) {
        tokens.emplace_back(std::move(current));
    }

    return tokens;
}

} // namespace sentiment
