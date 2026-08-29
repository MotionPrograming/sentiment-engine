#include "sentiment/text/vocabulary.hpp"
using namespace std;

namespace sentiment {

Vocabulary::TokenId
Vocabulary::add(std::string_view token) {

    auto it = token_to_id_.find(std::string(token));

    if (it != token_to_id_.end()) {
        return it->second;
    }

    const TokenId id = id_to_token_.size();

    id_to_token_.emplace_back(token);

    token_to_id_.emplace(
        id_to_token_.back(),
        id
    );

    return id;
}

Vocabulary::TokenId
Vocabulary::find(std::string_view token) const noexcept {

    auto it = token_to_id_.find(std::string(token));

    if (it == token_to_id_.end()) {
        return static_cast<TokenId>(-1);
    }

    return it->second;
}

bool Vocabulary::contains(std::string_view token) const noexcept {

    return token_to_id_.find(std::string(token))
        != token_to_id_.end();
}

std::size_t Vocabulary::size() const noexcept {

    return id_to_token_.size();
}

const std::string&
Vocabulary::token(TokenId id) const {

    return id_to_token_.at(id);
}

} // namespace sentiment
