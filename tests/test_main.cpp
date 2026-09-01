#include "TestCases.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        myplacement::test::runDensityFieldTests();
        myplacement::test::runDensityModelTests();
        myplacement::test::runPlacementFlowTests();
        std::cout << "All MyPlacement tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
