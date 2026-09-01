#pragma once

#include "sentiment/core/types.hpp"

#include <unordered_map>

namespace sentiment {

class Vocabulary {
public:
    using TokenId = u32;
    static constexpr TokenId InvalidId = static_cast<TokenId>(-1);

    TokenId add(sv token);
    [[nodiscard]] TokenId find(sv token) const noexcept;
    [[nodiscard]] bool contains(sv token) const noexcept;
    [[nodiscard]] sz size() const noexcept;
    [[nodiscard]] const str& token(TokenId id) const;

private:
    vec<str> id_to_token_;
    std::unordered_map<str, TokenId> token_to_id_;
};

} // namespace sentiment
