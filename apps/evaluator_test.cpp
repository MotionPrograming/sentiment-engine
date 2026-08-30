#include "sentiment/ml/evaluator.hpp"

#include <bits/stdc++.h>

using namespace std;

int main() {

    using namespace sentiment;

    cout
        << "========================================\n"
        << " SentimentEngine - Evaluator Test\n"
        << "========================================\n\n";

    const vec<Sentiment> actual = {

        Sentiment::Positive,
        Sentiment::Positive,
        Sentiment::Neutral,
        Sentiment::Neutral,
        Sentiment::Negative,
        Sentiment::Negative
    };

    const vec<Sentiment> predicted = {

        Sentiment::Positive,
        Sentiment::Neutral,
        Sentiment::Neutral,
        Sentiment::Neutral,
        Sentiment::Negative,
        Sentiment::Positive
    };

    const auto result =
        Evaluator::evaluate(
            actual,
            predicted
        );

    cout
        << fixed
        << setprecision(4);

    cout
        << "Accuracy    : "
        << result.accuracy
        << '\n';

    cout
        << "Macro F1    : "
        << result.macro_f1
        << '\n';

    cout
        << "Weighted F1 : "
        << result.weighted_f1
        << "\n\n";

    for (size_t c = 0; c < 3; ++c) {

        cout
            << "Class "
            << c
            << '\n';

        cout
            << "  Precision: "
            << result.precision[c]
            << '\n';

        cout
            << "  Recall   : "
            << result.recall[c]
            << '\n';

        cout
            << "  F1       : "
            << result.f1[c]
            << "\n\n";
    }

    cout
        << "Confusion Matrix\n"
        << "----------------\n";

    for (const auto& row :
         result.confusion) {

        for (const auto value :
             row) {

            cout
                << setw(5)
                << value
                << ' ';
        }

        cout << '\n';
    }

    cout
        << "\n========================================\n"
        << " Evaluator test completed successfully.\n"
        << "========================================\n";

    return 0;
}
