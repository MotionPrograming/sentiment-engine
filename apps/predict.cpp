
#include "sentiment/ml/linear_svm.hpp"
#include "sentiment/text/tfidf_vectorizer.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace std;
using namespace sentiment;

namespace {

static constexpr const char* Names[] = {
    "Negative",
    "Neutral",
    "Positive"
};

static constexpr size_t ClassCount = 3;


/*
 * ============================================================
 * Convert SVM decision scores to softmax confidence values.
 *
 * NOTE:
 * These are confidence scores, NOT calibrated probabilities.
 * Proper calibration requires a held-out calibration dataset.
 * ============================================================
 */
static vector<double>
softmax_confidence(
    const auto& scores
)
{
    vector<double> probabilities(
        ClassCount,
        0.0
    );

    double maximum_score =
        -numeric_limits<double>::infinity();

    for (
        size_t i = 0;
        i < ClassCount;
        ++i
    ) {

        maximum_score =
            max(
                maximum_score,
                scores[i]
            );
    }


    double sum =
        0.0;

    for (
        size_t i = 0;
        i < ClassCount;
        ++i
    ) {

        probabilities[i] =
            exp(
                scores[i] -
                maximum_score
            );

        sum +=
            probabilities[i];
    }


    if (
        sum > 0.0
    ) {

        for (
            double& probability :
            probabilities
        ) {

            probability /=
                sum;
        }
    }


    return probabilities;
}


/*
 * ============================================================
 * Confidence level
 * ============================================================
 */
static const char*
confidence_level(
    double confidence
)
{
    if (
        confidence >= 0.80
    ) {

        return "HIGH";
    }

    if (
        confidence >= 0.60
    ) {

        return "MEDIUM";
    }

    if (
        confidence >= 0.45
    ) {

        return "LOW";
    }

    return "UNCERTAIN";
}


/*
 * ============================================================
 * Production-style prediction policy
 * ============================================================
 */
static const char*
policy(
    double confidence,
    double margin
)
{
    if (
        confidence < 0.45
    ) {

        return "ABSTAIN";
    }

    if (
        confidence < 0.60 ||
        margin < 0.15
    ) {

        return "REVIEW";
    }

    return "ACCEPT";
}

} // namespace


int main(
    int argc,
    char* argv[]
)
{
    const string model_dir =
        argc > 1
            ? argv[1]
            : "../models";

    const string text =
        argc > 2
            ? argv[2]
            : "This product is excellent and I really love it";


    const string tfidf_path =
        model_dir +
        "/tfidf.model";

    const string svm_path =
        model_dir +
        "/svm.model";


    /*
     * ========================================================
     * Load TF-IDF
     * ========================================================
     */

    TfidfVectorizer vectorizer;

    if (
        !vectorizer.load(
            tfidf_path
        )
    ) {

        cerr
            << "ERROR: Cannot load TF-IDF model:\n"
            << tfidf_path
            << '\n';

        return 2;
    }


    /*
     * ========================================================
     * Load SVM
     *
     * LinearSVM::load() returns void.
     * Therefore do NOT use:
     *
     *     if (!svm.load(...))
     * ========================================================
     */

    LinearSVM svm(1);

    try {

        svm.load(
            svm_path
        );

    } catch (
        const exception& e
    ) {

        cerr
            << "ERROR: Cannot load SVM model:\n"
            << svm_path
            << '\n'
            << "Reason: "
            << e.what()
            << '\n';

        return 2;
    }


    /*
     * ========================================================
     * Feature compatibility
     * ========================================================
     */

    if (
        svm.feature_count() !=
        vectorizer.vocabulary_size()
    ) {

        cerr
            << "ERROR: Model feature mismatch.\n"
            << "TF-IDF features : "
            << vectorizer.vocabulary_size()
            << '\n'
            << "SVM features    : "
            << svm.feature_count()
            << '\n';

        return 2;
    }


    /*
     * ========================================================
     * Transform input text
     * ========================================================
     */

    const SparseVector features =
        vectorizer.transform_sparse(
            text
        );


    if (
        features.empty()
    ) {

        cerr
            << "ERROR: No known features found "
               "in input text.\n";

        return 3;
    }


    /*
     * ========================================================
     * Predict
     *
     * LinearSVM::predict() returns Sentiment directly.
     * ========================================================
     */

    const Sentiment prediction =
        svm.predict(
            features
        );


    /*
     * ========================================================
     * Decision scores
     * ========================================================
     */

    const auto scores =
        svm.decision_scores(
            features
        );


    const size_t label =
        static_cast<size_t>(
            prediction
        );


    if (
        label >= ClassCount
    ) {

        cerr
            << "ERROR: Invalid predicted class: "
            << label
            << '\n';

        return 4;
    }


    /*
     * ========================================================
     * Convert scores to confidence distribution
     * ========================================================
     */

    const vector<double> probabilities =
        softmax_confidence(
            scores
        );


    const double confidence =
        probabilities[label];


    /*
     * ========================================================
     * Find best and second-best decision scores
     * ========================================================
     */

    double first =
        -numeric_limits<double>::infinity();

    double second =
        -numeric_limits<double>::infinity();


    for (
        size_t i = 0;
        i < ClassCount;
        ++i
    ) {

        const double score =
            scores[i];


        if (
            score > first
        ) {

            second =
                first;

            first =
                score;

        } else if (
            score > second
        ) {

            second =
                score;
        }
    }


    const double margin =
        first -
        second;


    /*
     * ========================================================
     * Output
     * ========================================================
     */

    cout
        << "========================================\n"
        << " SentimentEngine - Production Prediction\n"
        << "========================================\n";


    cout
        << "Text       : "
        << text
        << '\n';


    cout
        << "Prediction : "
        << Names[label]
        << '\n';


    cout
        << "Score      : "
        << fixed
        << setprecision(6)
        << scores[label]
        << '\n';


    cout
        << "Confidence : "
        << fixed
        << setprecision(2)
        << confidence * 100.0
        << "%\n";


    cout
        << "Confidence level : "
        << confidence_level(
            confidence
        )
        << '\n';


    cout
        << "Decision margin  : "
        << fixed
        << setprecision(6)
        << margin
        << '\n';


    cout
        << "Policy           : "
        << policy(
            confidence,
            margin
        )
        << "\n\n";


    /*
     * ========================================================
     * Decision scores
     * ========================================================
     */

    cout
        << "Decision Scores\n";


    cout
        << "  Negative : "
        << fixed
        << setprecision(6)
        << scores[0]
        << '\n';


    cout
        << "  Neutral  : "
        << scores[1]
        << '\n';


    cout
        << "  Positive : "
        << scores[2]
        << "\n\n";


    /*
     * ========================================================
     * Confidence distribution
     * ========================================================
     *
     * These values are softmax-normalized SVM scores.
     *
     * They are useful as relative confidence indicators,
     * but they are NOT calibrated probabilities.
     * ========================================================
     */

    cout
        << "Confidence Distribution\n";


    cout
        << "  Negative : "
        << fixed
        << setprecision(4)
        << probabilities[0]
        << '\n';


    cout
        << "  Neutral  : "
        << probabilities[1]
        << '\n';


    cout
        << "  Positive : "
        << probabilities[2]
        << '\n';


    return 0;
}
