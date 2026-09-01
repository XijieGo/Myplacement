#pragma once

#include <filesystem>
#include <stdexcept>

#include "myplacement/model/PlacementDatabase.hpp"

namespace myplacement {

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class BookshelfParser {
public:
    PlacementDatabase parseAux(const std::filesystem::path& aux_path) const;
};

}  // namespace myplacement
