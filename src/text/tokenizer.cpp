#include "sentiment/text/tokenizer.hpp"

#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

bool Tokenizer::is_ascii_alpha(
    char c
) noexcept {

    return
        (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z');
}


char Tokenizer::to_lower(
    char c
) noexcept {

    if (
        c >= 'A' &&
        c <= 'Z'
    ) {

        return static_cast<char>(
            c + ('a' - 'A')
        );
    }

    return c;
}


vec<str>
Tokenizer::tokenize(
    sv text
) const {

    vec<str> words;

    words.reserve(
        text.size() / 5 + 4
    );

    str current;

    current.reserve(24);

    for (
        char c :
        text
    ) {

        if (
            is_ascii_alpha(c)
        ) {

            current.push_back(
                to_lower(c)
            );
        }
        else {

            if (
                !current.empty()
            ) {

                words.emplace_back(
                    move(current)
                );

                current.clear();

                current.reserve(24);
            }
        }
    }

    if (
        !current.empty()
    ) {

        words.emplace_back(
            move(current)
        );
    }

    return words;
}

} // namespace sentiment