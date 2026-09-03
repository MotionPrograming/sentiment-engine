#pragma once

#include "sentiment/core/types.hpp"

#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

struct EvaluationResult {

    u64 total_samples{0};
    u64 correct_predictions{0};

    double accuracy{0.0};

    arr<double, 3> precision{};

    arr<double, 3> recall{};

    arr<double, 3> f1{};

    double macro_f1{0.0};

    double weighted_f1{0.0};

    arr<arr<u64, 3>, 3> confusion{};
};

class Evaluator {

public:

    static EvaluationResult evaluate(
        const vec<Sentiment>& actual,
        const vec<Sentiment>& predicted
    );
};

}
