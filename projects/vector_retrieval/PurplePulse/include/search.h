#pragma once

#include <cstddef>
#include <memory>

#include "vector_types.h"

SearchResults cpuExactSearch(const VectorDatabase& database,
                             const QuerySet& queries,
                             const SearchParams& params);

SearchResults gpuExactSearch(const VectorDatabase& database,
                             const QuerySet& queries,
                             const SearchParams& params,
                             SearchStats* stats = nullptr);

// 将数据库和查询工作区常驻 GPU，适合预热、重复测量和服务化查询。
class GpuExactSearchEngine {
 public:
  GpuExactSearchEngine(const VectorDatabase& database,
                       const QuerySet& initial_queries,
                       const SearchParams& params);
  ~GpuExactSearchEngine();

  GpuExactSearchEngine(const GpuExactSearchEngine&) = delete;
  GpuExactSearchEngine& operator=(const GpuExactSearchEngine&) = delete;
  GpuExactSearchEngine(GpuExactSearchEngine&&) noexcept;
  GpuExactSearchEngine& operator=(GpuExactSearchEngine&&) noexcept;

  SearchResults search(const QuerySet& queries, SearchStats* stats = nullptr);
  double databaseH2DMilliseconds() const;
  std::size_t deviceBytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool resultsMatch(const SearchResults& expected, const SearchResults& actual,
                  float absolute_tolerance, std::string* error_message);
