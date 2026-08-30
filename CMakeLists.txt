cmake_minimum_required(VERSION 3.20)

project(
    SentimentEngine
    VERSION 0.1.0
    LANGUAGES CXX
)

# ============================================================
# C++ Configuration
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Dependencies
# ============================================================

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# ============================================================
# Core Library
# ============================================================

add_library(
    sentiment_core

    src/text/tokenizer.cpp
    src/text/vocabulary.cpp
    src/text/tfidf_vectorizer.cpp

    src/math/sparse_matrix_csr.cpp

    src/ml/linear_svm.cpp
    src/ml/evaluator.cpp

    src/io/dataset_reader.cpp
)

target_include_directories(
    sentiment_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    sentiment_core
    PUBLIC
        Arrow::arrow_shared
        Parquet::parquet_shared
)

target_compile_options(
    sentiment_core
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# ============================================================
# Prediction Application
# ============================================================

add_executable(
    sentiment_predict
    apps/predict.cpp
)

target_link_libraries(
    sentiment_predict
    PRIVATE
        sentiment_core
)

# ============================================================
# Training Application
# ============================================================

add_executable(
    sentiment_train
    apps/train.cpp
)

target_link_libraries(
    sentiment_train
    PRIVATE
        sentiment_core
)

# ============================================================
# Build Information
# ============================================================

message(STATUS "========================================")
message(STATUS " SentimentEngine")
message(STATUS " C++ Standard : C++20")
message(STATUS " Arrow        : ${Arrow_VERSION}")
message(STATUS "========================================")

# ============================================================
# Evaluator Test
# ============================================================

add_executable(
    sentiment_evaluator_test
    apps/evaluator_test.cpp
)

target_link_libraries(
    sentiment_evaluator_test
    PRIVATE
        sentiment_core
)
