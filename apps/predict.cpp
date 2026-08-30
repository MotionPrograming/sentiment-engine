#include "sentiment/text/tfidf_vectorizer.hpp"
#include "sentiment/ml/linear_svm.hpp"

#include <iomanip>
#include <iostream>

int main() {

    using namespace sentiment;

    // --------------------------------------------------------
    // Training documents
    // --------------------------------------------------------

    const vec<str> documents = {
        "excellent great amazing",
        "great excellent good",
        "amazing wonderful excellent",

        "the product is okay",
        "average normal experience",
        "nothing special average",

        "terrible bad awful",
        "bad horrible experience",
        "awful terrible product"
    };

    const vec<Sentiment> labels = {
        Sentiment::Positive,
        Sentiment::Positive,
        Sentiment::Positive,

        Sentiment::Neutral,
        Sentiment::Neutral,
        Sentiment::Neutral,

        Sentiment::Negative,
        Sentiment::Negative,
        Sentiment::Negative
    };

    std::cout
        << "========================================\n"
        << " SentimentEngine - Linear SVM Test\n"
        << "========================================\n\n";

    // --------------------------------------------------------
    // TF-IDF
    // --------------------------------------------------------

    TfidfVectorizer vectorizer(100);

    std::cout << "Fitting TF-IDF...\n";

    vectorizer.fit(documents);

    const sz feature_count =
        vectorizer.vocabulary_size();

    std::cout
        << "Vocabulary size: "
        << feature_count
        << "\n\n";

    // --------------------------------------------------------
    // Transform training data
    // --------------------------------------------------------

    vec<vec<double>> features;

    features.reserve(documents.size());

    for (const auto& document : documents) {

        features.push_back(
            vectorizer.transform(document)
        );
    }

    std::cout
        << "Training vectors: "
        << features.size()
        << "\n\n";

    // --------------------------------------------------------
    // Train SVM
    // --------------------------------------------------------

    LinearSVM svm(feature_count);

    std::cout << "Training Linear SVM...\n";

    svm.train(
        features,
        labels,
        50,
        0.05,
        0.0001
    );

    std::cout
        << "SVM trained: "
        << std::boolalpha
        << svm.trained()
        << "\n\n";

    // --------------------------------------------------------
    // Training accuracy
    // --------------------------------------------------------

    sz correct = 0;

    std::cout << "Training predictions:\n\n";

    for (sz i = 0; i < documents.size(); ++i) {

        const Prediction prediction =
            svm.predict(features[i]);

        const bool is_correct =
            prediction.label == labels[i];

        if (is_correct) {
            ++correct;
        }

        std::cout
            << "[" << i + 1 << "] "
            << documents[i]
            << "\n";

        std::cout
            << "    expected: "
            << to_string(labels[i])
            << "\n";

        std::cout
            << "    predicted: "
            << to_string(prediction.label)
            << "\n";

        std::cout
            << "    score: "
            << std::fixed
            << std::setprecision(6)
            << prediction.score
            << "\n";

        std::cout
            << "    result: "
            << (is_correct ? "PASS" : "FAIL")
            << "\n\n";
    }

    const double accuracy =
        static_cast<double>(correct) /
        static_cast<double>(documents.size());

    std::cout
        << "Training accuracy: "
        << std::fixed
        << std::setprecision(2)
        << accuracy * 100.0
        << "%\n\n";

    // --------------------------------------------------------
    // Unseen prediction tests
    // --------------------------------------------------------

    const vec<str> test_documents = {
        "excellent wonderful product",
        "average product experience",
        "terrible horrible product"
    };

    std::cout
        << "Unseen prediction tests:\n\n";

    for (const auto& document : test_documents) {

        const auto test_features =
            vectorizer.transform(document);

        const Prediction prediction =
            svm.predict(test_features);

        std::cout
            << "Text: "
            << document
            << "\n";

        std::cout
            << "Predicted: "
            << to_string(prediction.label)
            << "\n";

        std::cout
            << "Score: "
            << std::fixed
            << std::setprecision(6)
            << prediction.score
            << "\n\n";
    }

    // --------------------------------------------------------
    // Final verification
    // --------------------------------------------------------

    if (!svm.trained()) {

        std::cerr
            << "ERROR: SVM was not marked as trained.\n";

        return 1;
    }

    if (accuracy < 0.80) {

        std::cerr
            << "ERROR: Training accuracy below 80%.\n";

        return 1;
    }

    std::cout
        << "========================================\n"
        << " Linear SVM test completed successfully.\n"
        << "========================================\n";

    return 0;
}
