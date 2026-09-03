#include "sentiment/text/vocabulary.hpp"

#include <utility>

namespace sentiment {

Vocabulary::TokenId
Vocabulary::add(sv token) {

    auto it = token_to_id_.find(std::string(token));

    if (it != token_to_id_.end()) {
        return it->second;
    }

    const TokenId id =
        static_cast<TokenId>(id_to_token_.size());

    id_to_token_.emplace_back(token);

    token_to_id_.emplace(
        id_to_token_.back(),
        id
    );

    return id;
}

Vocabulary::TokenId
Vocabulary::find(sv token) const noexcept {

    auto it = token_to_id_.find(std::string(token));

    if (it == token_to_id_.end()) {
        return InvalidId;
    }

    return it->second;
}

Vocabulary::TokenId
Vocabulary::find(const str& token) const noexcept {

    auto it = token_to_id_.find(token);

    if (it == token_to_id_.end()) {
        return InvalidId;
    }

    return it->second;
}

bool Vocabulary::contains(sv token) const noexcept {

    return token_to_id_.find(std::string(token))
        != token_to_id_.end();
}

sz Vocabulary::size() const noexcept {
    return id_to_token_.size();
}

bool Vocabulary::empty() const noexcept {
    return id_to_token_.empty();
}

const str&
Vocabulary::token(TokenId id) const {

    return id_to_token_.at(id);
}

const str&
Vocabulary::term(TokenId id) const {

    return token(id);
}

} // namespace sentiment