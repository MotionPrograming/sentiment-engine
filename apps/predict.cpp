#include "sentiment/text/tfidf_vectorizer.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {

    using namespace sentiment;

    const vec<str> documents = {
        "this app is great",
        "this app is excellent",
        "this app is crashing",
        "the application keeps crashing",
        "great application and excellent experience"
    };

    std::cout << "========================================\n";
    std::cout << " SentimentEngine - TF-IDF Test\n";
    std::cout << "========================================\n\n";

    TfidfVectorizer vectorizer(100);

    std::cout << "Fitting TF-IDF...\n";

    vectorizer.fit(documents);

    std::cout
        << "Vocabulary size: "
        << vectorizer.vocabulary_size()
        << "\n\n";

    const str test_document =
        "this app is crashing";

    const auto features =
        vectorizer.transform(test_document);

    std::cout
        << "Document: "
        << test_document
        << "\n\n";

    std::cout << "TF-IDF vector:\n";

    for (sz i = 0; i < features.size(); ++i) {

        if (features[i] > 0.0) {

            std::cout
                << "feature[" << i << "] = "
                << std::fixed
                << std::setprecision(6)
                << features[i]
                << '\n';
        }
    }

    double norm = 0.0;

    for (const auto value : features) {
        norm += value * value;
    }

    std::cout
        << "\nL2 norm: "
        << std::sqrt(norm)
        << '\n';

    std::cout
        << "\nTF-IDF test completed successfully.\n";

    return 0;
}
