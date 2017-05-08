//
// Created by Seth Parker on 7/29/16.
//

#define BOOST_TEST_MODULE ClothModelingUVMapping

#include <boost/test/unit_test.hpp>
#include "vc/core/shapes/Arch.hpp"
#include "vc/core/vc_defines.hpp"
#include "vc/testing/ParsingHelpers.hpp"
#include "vc/testing/TestingUtils.hpp"
#include "vc/texturing/ClothModelingUVMapping.hpp"

using namespace volcart;

/***************************************************************************************
 *                                                                                     *
 *  clothTest.cpp - tests the functionality of
 * /v-c/texturing/clothModelingUV.cpp      *
 *  The ultimate goal of this file is the following: *
 *                                                                                     *
 *    Confirm that the class produces UV coordinates that match expected output.
 * *
 *                                                                                     *
 *  This file is broken up into a test fixture which initializes the objects *
 *  used in each of the test cases. *
 *                                                                                     *
 *   1. ArchClothUVTest *
 *                                                                                     *
 * Input: *
 *     No required inputs for this sample test. All objects needed for the test
 * cases  *
 *     are constructed and destroyed by the fixtures. Note, we're only testing
 * the     *
 *     non-enclosed shapes here. Sphere, Cone and Cube have been omitted.
 * Pregenerated *
 *     test files were generated by vc_clothExample. *
 *                                                                                     *
 * Test-Specific Output: *
 *     Specific test output only given on failure of any tests. Otherwise,
 * general     *
 *     number of testing errors is output. *
 *                                                                                     *
 * *************************************************************************************/

/*
 *
 *    FIXTURES
 *
 */

struct CreateArchClothUVFixture {

    CreateArchClothUVFixture()
    {

        std::cerr << "Creating Cloth Modeling - Arch UV map..." << std::endl;

        // get ITK Mesh
        _in_Mesh = _Arch.itkMesh();

        // Setup params
        _unfurlIt = 20000;
        _collisionIt = 0;
        _expansionIt = 5000;

        _unfurlPins.push_back(0);
        _unfurlPins.push_back(90);

        // Setup the simulation
        volcart::texturing::ClothModelingUVMapping clothUV(
            _in_Mesh, _unfurlIt, _collisionIt, _expansionIt, _unfurlPins,
            _expandPins);
        clothUV.setAcceleration(
            volcart::texturing::ClothModelingUVMapping::Stage::Unfurl, 10);
        clothUV.setAcceleration(
            volcart::texturing::ClothModelingUVMapping::Stage::Collision, -10);
        clothUV.setAcceleration(
            volcart::texturing::ClothModelingUVMapping::Stage::Expansion, -10);

        // Run the simulation in parts
        clothUV.unfurl();
        _out_Mesh_unfurl = clothUV.getMesh();
        clothUV.collide();
        _out_Mesh_collide = clothUV.getMesh();
        clothUV.expand();
        _out_Mesh_final = clothUV.getMesh();

        // Load pre-generated output from file
        volcart::testing::ParsingHelpers::ParseOBJFile(
            "clothUV_Arch_Unfurl.obj", _SavedPoints_unfurl, _SavedCells_unfurl);
        volcart::testing::ParsingHelpers::ParseOBJFile(
            "clothUV_Arch_Collide.obj", _SavedPoints_collide,
            _SavedCells_collide);
        volcart::testing::ParsingHelpers::ParseOBJFile(
            "clothUV_Arch_Final.obj", _SavedPoints_final, _SavedCells_final);
    }

    ~CreateArchClothUVFixture()
    {
        std::cerr << "Destroying Cloth Modeling - Arch UV map..." << std::endl;
    }

    // declare Arch mesh
    volcart::shapes::Arch _Arch;
    ITKMesh::Pointer _in_Mesh;
    ITKMesh::Pointer _out_Mesh_unfurl, _out_Mesh_collide, _out_Mesh_final;

    // Simulation
    volcart::texturing::ClothModelingUVMapping::VertIDList _unfurlPins;
    volcart::texturing::ClothModelingUVMapping::VertIDList _expandPins;
    uint16_t _unfurlIt;
    uint16_t _collisionIt;
    uint16_t _expansionIt;

    std::vector<Vertex> _SavedPoints_unfurl;
    std::vector<Cell> _SavedCells_unfurl;

    std::vector<Vertex> _SavedPoints_collide;
    std::vector<Cell> _SavedCells_collide;

    std::vector<Vertex> _SavedPoints_final;
    std::vector<Cell> _SavedCells_final;
};

/*
 *
 *    TEST CASES
 *
 */

BOOST_FIXTURE_TEST_CASE(ArchClothUVTest, CreateArchClothUVFixture)
{

    ///// Unfurl /////
    std::cerr << "Comparing results of unfurling step..." << std::endl;
    // check size of uvMap and number of points in mesh
    BOOST_CHECK_EQUAL(
        _out_Mesh_unfurl->GetNumberOfPoints(), _in_Mesh->GetNumberOfPoints());
    BOOST_CHECK_EQUAL(
        _out_Mesh_unfurl->GetNumberOfPoints(), _SavedPoints_unfurl.size());

    // check uvmap against original mesh input pointIDs
    for (size_t point = 0; point < _SavedPoints_unfurl.size(); ++point) {
        volcart::testing::SmallOrClose(
            _out_Mesh_unfurl->GetPoint(point)[0], _SavedPoints_unfurl[point].x,
            1.0, 5.0);
        volcart::testing::SmallOrClose(
            _out_Mesh_unfurl->GetPoint(point)[1], _SavedPoints_unfurl[point].y,
            1.0, 5.0);
        volcart::testing::SmallOrClose(
            _out_Mesh_unfurl->GetPoint(point)[2], _SavedPoints_unfurl[point].z,
            1.0, 5.0);
    }

    ///// Collide /////
    std::cerr << "Comparing results of collision step..." << std::endl;
    // check size of uvMap and number of points in mesh
    BOOST_CHECK_EQUAL(
        _out_Mesh_collide->GetNumberOfPoints(), _in_Mesh->GetNumberOfPoints());
    BOOST_CHECK_EQUAL(
        _out_Mesh_collide->GetNumberOfPoints(), _SavedPoints_collide.size());

    // check uvmap against original mesh input pointIDs
    for (size_t point = 0; point < _SavedPoints_collide.size(); ++point) {
        volcart::testing::SmallOrClose(
            _out_Mesh_collide->GetPoint(point)[0],
            _SavedPoints_collide[point].x, 1.0, 5.0);
        volcart::testing::SmallOrClose(
            _out_Mesh_collide->GetPoint(point)[1],
            _SavedPoints_collide[point].y, 1.0, 5.0);
        volcart::testing::SmallOrClose(
            _out_Mesh_collide->GetPoint(point)[2],
            _SavedPoints_collide[point].z, 1.0, 5.0);
    }

    ///// Expand/Final /////
    std::cerr << "Comparing results of unfurling step..." << std::endl;
    // check size of uvMap and number of points in mesh
    BOOST_CHECK_EQUAL(
        _out_Mesh_final->GetNumberOfPoints(), _in_Mesh->GetNumberOfPoints());
    BOOST_CHECK_EQUAL(
        _out_Mesh_final->GetNumberOfPoints(), _SavedPoints_final.size());

    // check uvmap against original mesh input pointIDs
    for (size_t point = 0; point < _SavedPoints_final.size(); ++point) {
        volcart::testing::SmallOrClose(
            _out_Mesh_final->GetPoint(point)[0], _SavedPoints_final[point].x,
            1.0, 5.0);
        volcart::testing::SmallOrClose(
            _out_Mesh_final->GetPoint(point)[1], _SavedPoints_final[point].y,
            1.0, 5.0);
        volcart::testing::SmallOrClose(
            _out_Mesh_final->GetPoint(point)[2], _SavedPoints_final[point].z,
            1.0, 5.0);
    }
}
