#pragma once

#include "sentiment/core/types.hpp"

#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sentiment {

class Vocabulary {

public:

    using TokenId = sz;

    static constexpr TokenId InvalidTokenId =
        std::numeric_limits<TokenId>::max();

    TokenId add(sv token);

    [[nodiscard]]
    TokenId find(sv token) const noexcept;

    [[nodiscard]]
    bool contains(sv token) const noexcept;

    [[nodiscard]]
    sz size() const noexcept;

    [[nodiscard]]
    const str& token(TokenId id) const;

private:

    std::unordered_map<str, TokenId> token_to_id_;

    vec<str> id_to_token_;
};

} // namespace sentiment
