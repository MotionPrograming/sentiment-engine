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

    current.reserve(32);

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
                current.reserve(32);
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

    if (
        words.empty()
    ) {

        return {};
    }


    /*
     * ========================================================
     * Unigram + bigram generation
     *
     * Example:
     *
     * "this product works"
     *
     * becomes:
     *
     * this
     * product
     * works
     * this_product
     * product_works
     * ========================================================
     */

    const sz word_count =
        words.size();

    vec<str> tokens;

    tokens.reserve(
        word_count +
        (word_count > 1
            ? word_count - 1
            : 0)
    );

    for (
        const auto& word :
        words
    ) {

        tokens.push_back(
            word
        );
    }

    if (
        word_count > 1
    ) {

        for (
            sz i = 0;
            i + 1 < word_count;
            ++i
        ) {

            str bigram;

            bigram.reserve(
                words[i].size() +
                words[i + 1].size() +
                1
            );

            bigram.append(
                words[i]
            );

            bigram.push_back('_');

            bigram.append(
                words[i + 1]
            );

            tokens.emplace_back(
                move(bigram)
            );
        }
    }

    return tokens;
}

} // namespace sentiment